from typing import Optional, Tuple
import math
import torch
from torch import nn
import torch.distributed as dist

from .fused_ops import ema_parameters, ema_hidden, fftconv
from megalodon.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from megalodon.distributed.utils import divide_and_check_no_remainder

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def _reset_parameters(alpha, delta, theta, gamma, omega, embed_dim):
    # delta & alpha
    nn.init.normal_(alpha, mean=0.0, std=0.2)
    nn.init.normal_(delta, mean=0.0, std=0.2)
    # sync global permuted index
    idx = torch.randperm(embed_dim)
    idx_ = idx.cuda()
    dist.broadcast(idx_, src=0)
    idx = idx_.to(idx.device)
    # theta
    freqs = math.log(embed_dim) / embed_dim
    freqs = torch.exp(torch.arange(1, embed_dim + 1, requires_grad=False) * -freqs)
    freqs = freqs[idx]
    freqs = freqs.to(theta).view(embed_dim, 1, 1)
    freqs = torch.log(freqs / (1.0 - freqs))
    with torch.no_grad():
        theta.copy_(freqs)
    # gamma # omega
    nn.init.normal_(gamma, mean=0.0, std=1.0)
    with torch.no_grad():
        gamma[:, :, 1] = 0.
    nn.init.trunc_normal_(omega, mean=0.0, std=0.25, a=-1.0, b=1.0)


class MultiHeadComplexEMA(nn.Module):
    """Complex Exponential Moving Average Layer.

    See "Megalodon paper" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / self.ndim)

        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.dim_per_partition = divide_and_check_no_remainder(embed_dim, world_size)

        self.alpha = nn.Parameter(torch.Tensor(self.dim_per_partition, ndim, 1))
        self.delta = nn.Parameter(torch.Tensor(self.dim_per_partition, ndim, 1))
        self.theta = nn.Parameter(torch.Tensor(self.dim_per_partition, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(self.dim_per_partition, ndim, 2))
        self.omega = nn.Parameter(torch.Tensor(self.dim_per_partition, 1))
        self._coeffs = None
        # init parameters
        self._init_parameters()

    def _init_parameters(self):
        world_size = get_model_parallel_world_size()
        if world_size == 1:
            _reset_parameters(self.alpha, self.delta, self.theta, self.gamma, self.omega, self.embed_dim)
            return None

        master_alpha = torch.empty(self.embed_dim, self.ndim, 1, dtype=self.alpha.dtype, requires_grad=False)
        master_delta = torch.empty(self.embed_dim, self.ndim, 1, dtype=self.delta.dtype, requires_grad=False)
        master_theta = torch.empty(self.embed_dim, 1, 1, dtype=self.theta.dtype, requires_grad=False)
        master_gamma = torch.empty(self.embed_dim, self.ndim, 2, dtype=self.gamma.dtype, requires_grad=False)
        master_omega = torch.empty(self.embed_dim, 1, dtype=self.omega.dtype, requires_grad=False)

        _reset_parameters(master_alpha, master_delta, master_theta, master_gamma, master_omega, self.embed_dim)
        rank = get_model_parallel_rank()
        my_alpha = torch.split(master_alpha, self.dim_per_partition, dim=0)[rank]
        my_delta = torch.split(master_delta, self.dim_per_partition, dim=0)[rank]
        my_theta = torch.split(master_theta, self.dim_per_partition, dim=0)[rank]
        my_gamma = torch.split(master_gamma, self.dim_per_partition, dim=0)[rank]
        my_omega = torch.split(master_omega, self.dim_per_partition, dim=0)[rank]

        with torch.no_grad():
            self.alpha.copy_(my_alpha)
            self.delta.copy_(my_delta)
            self.theta.copy_(my_theta)
            self.gamma.copy_(my_gamma)
            self.omega.copy_(my_omega)

        return None

    def _calc_coeffs(self):
        self._coeffs = None
        # D x 1 x 1
        theta = torch.sigmoid(self.theta.float()) * (2 * math.pi / self.ndim)
        # 1 x N
        wavelets = torch.arange(1, self.ndim + 1, dtype=theta.dtype, device=theta.device).view(1, self.ndim)
        # D x N x 1
        theta = wavelets.unsqueeze(2) * theta

        # D x N x 1
        alpha = torch.sigmoid(self.alpha.float())
        delta = torch.sigmoid(self.delta.float())
        # coeffs
        p = alpha
        q = torch.polar(1.0 - alpha * delta, theta)
        # D x N
        gamma = _r2c(self.gamma.float()) * self.scale
        return p, q, gamma

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor],
        compute_last_state: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # B x D x L
        bsz, _, seq_len = x.size()
        residual = x * self.omega

        p, q, gamma = self.coeffs()
        k, b = ema_parameters(p, q, gamma, hx, seq_len)
        output = fftconv(x, k)
        if b is not None:
            output = output + b.to(output)
        # compute the last hidden state
        h = ema_hidden(x, p, q, hx) if compute_last_state else None
        # residual
        output = output + residual
        return output, h

    def extra_repr(self) -> str:
        return 'edim={} ({}), ndim={}'.format(self.embed_dim, self.dim_per_partition, self.ndim)
