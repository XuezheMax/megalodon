import torch
from torch import nn
import torch.nn.functional as F

from megalodon.modules.model_parallel import (
    RowParallelLinear,
    ColumnParallelLinear,
    gather_from_model_parallel_region,
)
from .fused_ops import memory_efficient_dropout
from .layer_norm import FusedLayerNorm
from megalodon.utils import get_init_fn


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        model_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        swiglu=False,
        norm_affine=True,
        norm_eps=1e-5,
        rescale=None,
        init_mode='bert',
    ):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout
        self.swiglu = swiglu
        self.rescale_init = rescale
        self.init_mode = init_mode

        self.norm = FusedLayerNorm(model_dim, elementwise_affine=norm_affine, eps=norm_eps)

        # layers
        self.fc1 = ColumnParallelLinear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            input_is_parallel=False,
            gather_output=False,
            init_method=get_init_fn(init_mode),
        )
        self.fc2 = RowParallelLinear(
            ffn_hidden_dim,
            model_dim,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=get_init_fn(init_mode),
        )
        self.fc3 = ColumnParallelLinear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            input_is_parallel=False,
            gather_output=False,
            init_method=get_init_fn(init_mode),
        ) if self.swiglu else None

        if rescale is None:
            self.register_parameter('alpha', None)
        else:
            assert rescale > 0., 'Layer scale init value should be positive.'
            self.alpha = nn.Parameter(torch.full((model_dim,), rescale))

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.alpha is None else (self.alpha * x)

    def forward(self, x, residual):
        # B x L x D
        x_ = gather_from_model_parallel_region(x)
        x = self.norm(x_)

        # fc1 & fc3
        if self.swiglu:
            hidden = F.silu(self.fc1(x)) * self.fc3(x)
            hidden = memory_efficient_dropout(hidden, self.hidden_dropout, self.training)
        else:
            hidden = F.silu(self.fc1(x))
            hidden = memory_efficient_dropout(hidden, self.hidden_dropout, self.training)

        # fc2
        x = self.fc2(hidden)
        x = memory_efficient_dropout(x, self.dropout, self.training)
        # residual
        out = self.rescale(x) + residual

        return out

    def extra_repr(self) -> str:
        return 'dim={}, hdim={}, swiglu={}, init={}, rescale={}'.format(self.model_dim, self.hidden_dim, self.swiglu,
                                                                        self.init_mode, self.rescale_init)
