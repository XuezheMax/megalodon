from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from .rotary_positional_embedding import apply_rotary_emb
from .layer_norm import FusedRMSNorm
from .timestep_norm import TimestepNorm
from .complex_exponential_moving_average import MultiHeadComplexEMA
from megalodon.distributed import (
    get_model_parallel_world_size,
    get_chunk_parallel_rank,
)
from megalodon.distributed.utils import divide_and_check_no_remainder
from .model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_from_model_parallel_region,
)
from .chunk_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_chunk_parallel_region,
    recv_from_prev_chunk_parallel_region,
)

from .fused_ops import swift_efficient_attention, memory_efficient_dropout
from megalodon.utils import get_init_fn

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def get_efficient_attention_function(attn_name: str):
    return {
        "swift": swift_efficient_attention,
        "fused": swift_efficient_attention
    }[attn_name]


class _InnerAttention(nn.Module):
    """
    Inner attention
    """

    def __init__(
        self,
        z_head_dim: int,
        v_head_dim: int,
        n_heads: int,
        chunk_size: int,
        dropout: float,
        efficient_attn: Optional[str],
    ):
        super().__init__()
        self.z_head_dim = z_head_dim
        self.v_head_dim = v_head_dim
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.efficient_attn = efficient_attn

        # efficient attention
        if self.efficient_attn is not None:
            self.attn_fn = get_efficient_attention_function(self.efficient_attn)
            assert self.attn_fn is not None
        else:
            self.attn_fn = None

    def forward(
        self,
        xq: Tensor,
        xk: Tensor,
        xv: Tensor,
        mask: Optional[Tensor],
        freqs_cis: Optional[Tensor],
        cache: Optional[Tuple[Tensor, Tensor, int]] = None,
    ):
        bs, slen, _ = xq.shape
        xq = xq.view(bs, slen, self.n_heads, self.z_head_dim)
        xk = xk.view(bs, slen, self.n_heads, self.z_head_dim)
        xv = xv.view(bs, slen, self.n_heads, self.v_head_dim)

        # shorter than a chunk, apply rope before cat cache_kv
        if slen <= self.chunk_size:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, backward=False)

        new_cache = None
        clen = slen
        if cache is not None:
            cache_k, cache_v, count = cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)

            clen = count + slen
            new_count = clen % self.chunk_size
            cache_k = xk.detach()[:, (clen - new_count):]
            cache_v = xv.detach()[:, (clen - new_count):]
            new_cache = (cache_k, cache_v, new_count)

        if self.chunk_size < slen:
            nc = slen // self.chunk_size
            xq = xq.view(bs * nc, self.chunk_size, self.n_heads, self.z_head_dim)

        if self.chunk_size < clen:
            nc = clen // self.chunk_size
            xk = xk.view(bs * nc, self.chunk_size, self.n_heads, self.z_head_dim)
            xv = xv.view(bs * nc, self.chunk_size, self.n_heads, self.v_head_dim)

        # seq length > chunk, apply rope after chunk split
        if slen > self.chunk_size:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, backward=False)

        if self.efficient_attn is None:
            # B*K x H x C x S
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            assert mask is not None
            # B*K x H x C x C
            scores = torch.matmul(xq, xk.transpose(2, 3))
            scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq)
            scores = memory_efficient_dropout(scores, self.dropout, self.training)
            # B*K x H x C x S -> B*k x C x H x S
            output = torch.matmul(scores, xv)
            output = output.transpose(1, 2).contiguous()
        else:
            assert mask is None
            if slen == 1 and cache is not None:
                use_causal_mask = False
            else:
                use_causal_mask = True
            output = self.attn_fn(xq, xk, xv, 1.0, self.dropout, use_causal_mask, self.training)

        output = output.view(bs, slen, -1)
        return output, new_cache

    def extra_repr(self) -> str:
        return 'heads={}, z_head_dim={}, v_head_dim={}, chunk={}'.format(self.n_heads, self.z_head_dim, self.v_head_dim, self.chunk_size)


class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.
    See "" for more details.
    """

    def __init__(
        self,
        mdim: int,
        zdim: int,
        hdim: int,
        ndim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = 2048,
        efficient_attn: Optional[str] = None,
        norm_num_groups: Optional[int] = None,
        norm_affine: bool = True,
        norm_eps: float = 1e-5,
        init_mode: str = 'he',
    ):
        super().__init__()

        self.mdim = mdim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.num_heads = num_heads
        self.init_mode = init_mode
        assert zdim % num_heads == 0 and hdim % num_heads == 0
        self.z_head_dim = zdim // num_heads
        self.v_head_dim = hdim // num_heads

        # Divide the weight matrix along the last dimension.
        model_parallel_world_size = get_model_parallel_world_size()
        self.local_heads = divide_and_check_no_remainder(num_heads, model_parallel_world_size)
        self.local_mdim = divide_and_check_no_remainder(mdim, model_parallel_world_size)

        self.chunk_size = chunk_size
        self.efficient_attn = efficient_attn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self.timenorm = TimestepNorm(mdim, norm_num_groups, eps=norm_eps)
        self.cema = MultiHeadComplexEMA(mdim, ndim)
        self.rmsnorm = FusedRMSNorm(mdim, elementwise_affine=norm_affine, eps=norm_eps)
        self.znorm = FusedRMSNorm(self.z_head_dim, elementwise_affine=False, eps=norm_eps)

        init_fn = get_init_fn(init_mode)
        self.wv = ColumnParallelLinear(
            mdim,
            hdim,
            bias=True,
            input_is_parallel=True,
            gather_output=False,
            init_method=init_fn
        )
        self.wz = ColumnParallelLinear(
            mdim,
            zdim,
            bias=True,
            input_is_parallel=False,
            gather_output=False,
            init_method=init_fn
        )
        self.wr = ColumnParallelLinear(
            mdim,
            hdim,
            bias=True,
            input_is_parallel=False,
            gather_output=False,
            init_method=init_fn
        )
        self.wh1 = ColumnParallelLinear(
            mdim,
            mdim,
            bias=True,
            input_is_parallel=False,
            gather_output=False,
            init_method=init_fn
        )
        self.wh2 = RowParallelLinear(
            hdim,
            mdim,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=init_fn
        )
        self.inner_attention = _InnerAttention(
            self.z_head_dim,
            self.v_head_dim,
            self.local_heads,
            self.chunk_size,
            self.attention_dropout,
            efficient_attn
        )
        self.gamma = Parameter(torch.zeros(2, self.z_head_dim * self.local_heads))
        self.beta = Parameter(torch.zeros(2, self.z_head_dim * self.local_heads))

    def _create_empty_prev_tensors(self, x):
        bsz, seq_len, _ = x.size()
        chunk_rank = get_chunk_parallel_rank()
        n_groups = self.timenorm.groups_per_partition
        ndim = self.cema.ndim
        dim = self.local_mdim
        prev_count = torch.full((bsz,), seq_len * chunk_rank, dtype=torch.int64, device=x.device)
        prev_tensor = torch.empty((bsz, n_groups * 2 + dim * ndim * 2), dtype=torch.float32, device=x.device, requires_grad=self.training)
        return prev_count, prev_tensor

    def _pack_prev_tensors(self, prev_mean, prev_var, hx):
        # B x D x N x 2 -> B x (D*N*2)
        h = _c2r(hx).flatten(1)
        # B x (G*2+D*N*2)
        prev_tensor = torch.cat([prev_mean, prev_var, h], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, x, prev_tensor):
        bsz = x.size(0)
        n_groups = self.timenorm.groups_per_partition
        ndim = self.cema.ndim
        dim = self.local_mdim
        prev_mean, prev_var, hx = torch.split(prev_tensor, [n_groups, n_groups, 2 * dim * ndim], dim=-1)
        prev_mean = prev_mean.to(x)
        prev_var = prev_var.to(x)
        hx = _r2c(hx.view(bsz, dim, ndim, 2))
        return prev_mean, prev_var, hx

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor], Tensor]] = None,
    ):
        bsz, seq_len, _ = x.size()
        residual = x

        if cache is not None:
            cache_attn, cache_norm, hx = cache
            prev_count, prev_mean, prev_var = cache_norm
        elif should_recv_from_prev():
            prev_count, prev_tensor = self._create_empty_prev_tensors(x)
            prev_tensor = recv_from_prev_chunk_parallel_region(prev_tensor)
            prev_mean, prev_var, hx = self._unpack_prev_tensors(x, prev_tensor)
            cache_attn, cache_norm = None, None
        else:
            prev_count, prev_mean, prev_var = None, None, None
            cache_attn, cache_norm, hx = None, None, None

        # B x L x D
        out_tsn, prev_count, prev_mean, prev_var = self.timenorm(x, prev_count, prev_mean, prev_var)
        # B x D x L
        compute_h = cache_attn is not None or should_send_to_next()
        out_cema, hx = self.cema(out_tsn.transpose(1, 2), hx, compute_last_state=compute_h)

        if cache is not None:
            cache_norm = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
            hx = None if hx is None else hx.detach()
        elif should_send_to_next():
            prev_tensor = self._pack_prev_tensors(prev_mean, prev_var, hx)
            prev_tensor = send_to_next_chunk_parallel_region(prev_tensor)
            # TODO: more elegent solution to force call bwd of prev_tensor
            out_cema = out_cema + prev_tensor.to(out_cema).mean() * 0

        out_cema = gather_from_model_parallel_region(out_cema, dim=1)
        # B x L x D
        out_cema = out_cema.transpose(1, 2)
        mx = self.rmsnorm(out_cema)
        mx = memory_efficient_dropout(mx, self.hidden_dropout, self.training)

        # B x L x S
        z = self.wz(mx)
        # B x L x S -> B x L x H x S/H
        z = z.view(bsz, seq_len, self.local_heads, self.z_head_dim)
        # B x L x H x S/H -> B x L x S
        z = self.znorm(z).view(bsz, seq_len, -1)
        # B x L x S -> B x L x 1 x S -> B x L x 2 x S
        gamma = (self.gamma + 1.0) / math.sqrt(self.z_head_dim)
        z = z.unsqueeze(2) * gamma + self.beta
        # B x L x 2 x S -> B x L x S
        q, k = torch.unbind(z, dim=2)

        # B x L x E
        v = F.silu(self.wv(out_tsn))
        r = F.silu(self.wr(mx))

        # B x L x E
        attn, cache_attn = self.inner_attention(q, k, v, mask, freqs_cis, cache_attn)
        attn = memory_efficient_dropout(attn * r, self.hidden_dropout, self.training)

        # B x L x E -> B x L x D
        h = self.wh1(mx) + self.wh2(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)

        # B x L x D
        out = h + residual

        if cache is not None:
            cache = (cache_attn, cache_norm, hx)

        return out, cache

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, hdim={}, heads={}, chunk={}, eff_attn={}, init={}'.format(self.mdim, self.zdim, self.hdim,
                                                                                            self.num_heads, self.chunk_size,
                                                                                            self.efficient_attn, self.init_mode)
