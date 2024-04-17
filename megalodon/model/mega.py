from typing import Optional, Tuple, List
import math
import torch
from torch import Tensor
from torch import nn
from fairscale.nn.checkpoint import checkpoint_wrapper
from fairscale.nn.wrap.auto_wrap import wrap, enable_wrap

from megalodon.modules import (
    MovingAverageGatedAttention,
    NormalizedFeedForwardNetwork,
    RotaryEmbedding,
    TimestepNorm,
)
from megalodon.modules.fused_ops import memory_efficient_dropout
from megalodon.modules.model_parallel import (
    ParallelEmbedding,
    RowParallelLinear,
)
from megalodon.distributed import FullyShardedDataParallel
from megalodon.config import ModelConf
from megalodon.distributed import (
    get_chunk_parallel_world_size,
    get_chunk_parallel_rank,
    get_chunk_parallel_group,
    get_data_parallel_group,
    get_model_parallel_group,
    get_model_parallel_world_size,
)
from megalodon.modules.chunk_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_chunk_parallel_region,
    recv_from_prev_chunk_parallel_region,
)
from megalodon.utils import (
    get_torch_dtype,
    get_init_fn,
    create_on_gpu,
)


class MegaBlock(nn.Module):
    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()

        self.layer_id = layer_id

        self.mega = MovingAverageGatedAttention(
            mdim=cfg.model_dim,
            zdim=cfg.z_dim,
            hdim=cfg.value_dim,
            num_heads=cfg.num_heads,
            ndim=cfg.cema_ndim,
            chunk_size=cfg.chunk_size,
            efficient_attn=cfg.efficient_attn,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.norm_num_groups,
            norm_affine=cfg.norm_affine,
            norm_eps=cfg.norm_eps,
            init_mode=cfg.init_mode
        )

        rescale = 0.1 * (0.5 ** layer_id) if cfg.rescale_nffn else None
        self.nffn = NormalizedFeedForwardNetwork(
            model_dim=cfg.model_dim,
            ffn_hidden_dim=cfg.ffn_hidden_dim,
            swiglu=cfg.swiglu,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_affine=cfg.norm_affine,
            norm_eps=cfg.norm_eps,
            rescale=rescale,
            init_mode=cfg.init_mode
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor], Tensor]] = None,
    ):
        y, cache = self.mega(x, freqs_cis, mask, cache)
        out = self.nffn(y, x)
        return out, cache


class MegaOutputLayer(nn.Module):
    def __init__(self, cfg: ModelConf, embed_weight: nn.Parameter):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.output_size = cfg.vocab_size if cfg.output_size == -1 else cfg.output_size
        init_fn = get_init_fn('gaussian', dim=self.model_dim)
        self.final_norm = TimestepNorm(self.model_dim, cfg.norm_num_groups, eps=cfg.norm_eps)
        self.output = RowParallelLinear(self.model_dim, self.output_size, bias=False,
                                        input_is_parallel=True, parallel_output=False,
                                        init_method=init_fn)
        self.share_emb = cfg.share_emb
        if self.share_emb:
            self.output.weight = embed_weight

        self.chunk_parallel_rank = get_chunk_parallel_rank()

    def _create_empty_prev_tensors(self, x):
        bsz, seq_len, _ = x.size()
        n_groups = self.final_norm.groups_per_partition
        prev_count = torch.full((bsz,), seq_len * self.chunk_parallel_rank, dtype=torch.int64, device=x.device)
        prev_tensor = torch.empty((bsz, n_groups * 2), dtype=x.dtype, device=x.device, requires_grad=self.training)
        return prev_count, prev_tensor

    def _pack_prev_tensors(self, prev_mean, prev_var):
        # B x (G*2)
        prev_tensor = torch.cat([prev_mean, prev_var], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, x, prev_tensor):
        n_groups = self.final_norm.groups_per_partition
        prev_mean, prev_var = torch.split(prev_tensor, [n_groups, n_groups], dim=-1)
        prev_mean = prev_mean
        prev_var = prev_var
        return prev_mean, prev_var

    def forward(
        self,
        x: Tensor,
        cache: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ):
        if cache is not None:
            prev_count, prev_mean, prev_var = cache
        elif should_recv_from_prev():
            prev_count, prev_tensor = self._create_empty_prev_tensors(x)
            prev_tensor = recv_from_prev_chunk_parallel_region(prev_tensor)
            prev_mean, prev_var = self._unpack_prev_tensors(x, prev_tensor)
        else:
            prev_count, prev_mean, prev_var = None, None, None

        x, prev_count, prev_mean, prev_var = self.final_norm(x, prev_count, prev_mean, prev_var)

        if cache is not None:
            cache = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
        elif should_send_to_next():
            prev_tensor = self._pack_prev_tensors(prev_mean, prev_var)
            prev_tensor = send_to_next_chunk_parallel_region(prev_tensor)
            # TODO: more elegent solution to force call bwd of prev_tensor
            x = x + prev_tensor.to(x).mean() * 0

        out = self.output(x)

        return out, cache


class Mega(nn.Module):
    def __init__(self, cfg: ModelConf):
        super().__init__()
        assert cfg.vocab_size > 0
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_layers
        self.model_dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        self.chunk_size = cfg.chunk_size
        self.output_size = self.vocab_size if cfg.output_size == -1 else cfg.output_size
        self.dropout = cfg.dropout
        self.efficient_attn = cfg.efficient_attn
        self.emb_scale = math.sqrt(cfg.model_dim) if cfg.scale_emb else None
        self.share_emb = cfg.share_emb

        self.chunk_parallel_size = get_chunk_parallel_world_size()
        self.chunk_parallel_rank = get_chunk_parallel_rank()

        init_fn = get_init_fn('gaussian', dim=self.model_dim)
        self.embed = ParallelEmbedding(self.vocab_size, self.model_dim, gather_output=False, init_method=init_fn)
        self.z_head_dim = cfg.z_dim // cfg.num_heads
        self.v_head_dim = cfg.value_dim // cfg.num_heads
        self.local_heads = self.num_heads // get_model_parallel_world_size()
        self.rope = RotaryEmbedding(self.z_head_dim, cfg.chunk_size, base=cfg.rope_base)

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            block = MegaBlock(cfg, layer_id)
            layer = checkpoint_wrapper(block) if cfg.layerwise_ckpt else block
            self.layers.append(wrap(layer))

        self.output = MegaOutputLayer(cfg, self.embed.weight)

    def forward(
        self,
        tokens: Tensor,
        cache: Optional[Tuple[List[Tuple[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor], torch.Tensor]],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
    ):

        bsz, seq_len = tokens.shape

        if cache is None:
            cache_layers, cache_output = None, None
            cache_len = 0
            assert seq_len % self.chunk_parallel_size == 0
            seq_len = seq_len // self.chunk_parallel_size
            # split tokens into chunks
            start = self.chunk_parallel_rank * seq_len
            end = (self.chunk_parallel_rank + 1) * seq_len
            tokens = tokens[:, start:end]
        else:
            cache_layers, cache_output = cache
            cache_len = 0 if cache_layers[0][0] is None else cache_layers[0][0][-1]
            assert self.chunk_parallel_size == 1, "inference mode does not support chunk parallel."
            if cache_len > 0:
                assert cache_len + seq_len <= self.chunk_size

        assert seq_len < self.chunk_size or seq_len % self.chunk_size == 0, '{}/{}'.format(seq_len, self.chunk_size)

        # embeddings
        emb = self.embed(tokens)
        if self.emb_scale is not None:
            emb = emb * self.emb_scale
        x = memory_efficient_dropout(emb, self.dropout, self.training)

        if self.efficient_attn:
            mask = None
        else:
            mask_size = min(seq_len, self.chunk_size)
            mask = torch.full((1, 1, mask_size, mask_size + cache_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=cache_len + 1).type_as(x)

        # rope freqs
        freq_cis = self.rope.get_freqs_cis(cache_len, min(cache_len + seq_len, self.chunk_size))

        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, freq_cis, mask, cache_layers[i] if cache else None)
            if cache is not None:
                cache_layers[i] = layer_cache

        out, cache_output = self.output(x, cache_output)

        if cache is not None:
            cache = (cache_layers, cache_output)

        return out.float(), cache


def build_model(
    model_cfg: ModelConf,
    dtype: str,
    fp32_reduce_scatter: bool,
    reshard_after_forward: bool,
) -> Mega:
    fsdp_cfg = {
        "process_group": get_data_parallel_group(),
        "process_group_reduce_scatter": get_data_parallel_group(),
        "model_parallel_process_group": get_model_parallel_group(),
        "chunk_parallel_process_group": get_chunk_parallel_group(),
        "compute_dtype": get_torch_dtype(dtype),
        "state_dict_device": torch.device("cpu"),
        "mixed_precision": dtype != "fp32",
        "flatten_parameters": True,
        "fp32_reduce_scatter": fp32_reduce_scatter,
        "reshard_after_forward": reshard_after_forward,
    }
    with create_on_gpu():
        with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp_cfg):
            model = Mega(model_cfg)
            model = wrap(model.cuda())
            model.train()

        return model


# register some models configurations
ModelStore = {}

ModelStore["mega200M"] = ModelConf(num_layers=12, model_dim=1024, num_heads=1, z_dim=256, value_dim=2048, ffn_hidden_dim=2560,
                                   chunk_size=2048, norm_num_groups=32)

ModelStore["mega1.3B"] = ModelConf(num_layers=24, model_dim=2048, num_heads=2, z_dim=512, value_dim=4096, ffn_hidden_dim=4864,
                                   chunk_size=2048, norm_num_groups=64)

ModelStore["mega1.3B_pg19"] = ModelConf(num_layers=24, model_dim=2048, num_heads=2, z_dim=512, value_dim=4096, ffn_hidden_dim=3584,
                                        chunk_size=2048, norm_num_groups=64, swiglu=True, scale_emb=True, share_emb=True)

ModelStore["mega7.1B"] = ModelConf(num_layers=32, model_dim=4096, num_heads=4, z_dim=1024, value_dim=8192, ffn_hidden_dim=11264,
                                   chunk_size=2048, norm_num_groups=64)

ModelStore["mega7.3B"] = ModelConf(num_layers=32, model_dim=4096, num_heads=4, z_dim=1024, value_dim=8192, ffn_hidden_dim=8192,
                                   chunk_size=2048, norm_num_groups=64, swiglu=True)

# fmt: on
