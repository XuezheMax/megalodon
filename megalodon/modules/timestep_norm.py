from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

import megalodon_extension.ops as megalodon_ops
from megalodon.distributed import get_model_parallel_world_size
from megalodon.distributed.utils import divide_and_check_no_remainder


class TimestepNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        prev_count: torch.Tensor,
        prev_mean: torch.Tensor,
        prev_var: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
        eps: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_groups is None:
            y, count, mean, var, cummean, cumrstd = megalodon_ops.timestep_norm_fwd(
                x, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, eps
            )
            ctx.save_for_backward(x, prev_mean, count, cummean, cumrstd, gamma, padding_mask)
        else:
            y, count, mean, var, cummean, cumrstd = megalodon_ops.group_timestep_norm_fwd(
                 x, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, eps
            )
            ctx.save_for_backward(x, prev_mean, count, cummean, cumrstd, gamma, padding_mask)
        ctx.num_groups = num_groups  # num_groups is not a torch.Tensor
        return y, count, mean, var

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
        _,
        mean_grad: torch.Tensor,
        var_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, None, None, None]:
        num_groups = ctx.num_groups
        if num_groups is None:
            x, prev_mean, count, cummean, cumrstd, gamma, padding_mask = ctx.saved_tensors
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = megalodon_ops.timestep_norm_bwd(
                y_grad, mean_grad, var_grad, x, prev_mean, count, cummean, cumrstd, gamma, padding_mask
            )
        else:
            x, prev_mean, count, cummean, cumrstd, gamma, padding_mask = ctx.saved_tensors
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = megalodon_ops.group_timestep_norm_bwd(
                 y_grad, mean_grad, var_grad, x, prev_mean, count, cummean, cumrstd, gamma, padding_mask, num_groups
            )
        return x_grad, None, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad, None, None, None


timestep_norm = TimestepNormFunc.apply


class TimestepNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_groups: Optional[int] = None,
        prior_count: int = 0,
        eps: float = 1e-5,
    ) -> None:

        super().__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        self._prior_count = prior_count

        if num_groups is None or num_groups == num_features:
            num_groups = num_features
            assert prior_count > 1
        else:
            assert self.num_features % num_groups == 0

        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.features_per_partition = divide_and_check_no_remainder(num_features, world_size)
        self.groups_per_partition = divide_and_check_no_remainder(num_groups, world_size)

        self.register_buffer("prior_count", torch.tensor(prior_count, dtype=torch.int64))
        if prior_count > 0:
            self.register_parameter("prior_mean", Parameter(torch.zeros(self.groups_per_partition)))
            self.register_parameter("prior_logv", Parameter(torch.zeros(self.groups_per_partition)))
        else:
            self.register_buffer("prior_mean", torch.zeros(self.groups_per_partition))
            self.register_buffer("prior_logv", torch.zeros(self.groups_per_partition))

        self.register_parameter("weight", Parameter(torch.zeros(self.features_per_partition)))
        self.register_parameter("bias", Parameter(torch.zeros(self.features_per_partition)))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        prev_count: Optional[torch.Tensor] = None,
        prev_mean: Optional[torch.Tensor] = None,
        prev_var: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = x.size(0)
        if prev_count is None:
            prev_count = self.prior_count.expand(batch_size).contiguous()
        if prev_mean is None:
            prev_mean = self.prior_mean.expand(batch_size, -1).contiguous()
        if prev_var is None:
            prev_var = self.prior_logv.exp().expand(batch_size, -1).contiguous()

        output = timestep_norm(x, prev_count, prev_mean, prev_var,
                               self.weight + 1.0, self.bias, padding_mask,
                               self.groups_per_partition, self.eps)
        return output

    def extra_repr(self) -> str:
        return 'num_features={num_features} ({features_per_partition}), ' \
               'num_groups={num_groups} ({groups_per_partition}), ' \
               'prior_count={_prior_count}, eps={eps}'.format(**self.__dict__)
