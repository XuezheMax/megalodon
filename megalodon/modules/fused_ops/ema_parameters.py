from typing import Optional, Tuple
import torch
from torch.autograd.function import FunctionCtx

import megalodon_extension.ops as megalodon_ops


class EMAParametersFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        p: torch.Tensor,
        q: torch.Tensor,
        gamma: torch.Tensor,
        hx: Optional[torch.Tensor],
        length: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        log_q = q.log()
        weight, bias, vander = megalodon_ops.ema_parameters_fwd(p, log_q, gamma, hx, length)
        ctx.save_for_backward(p, log_q, gamma, hx, vander)
        return weight, bias

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        weight_grad: torch.Tensor,
        bias_grad: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], None]:
        p, log_q, gamma, hx, vander = ctx.saved_tensors
        p_grad, q_grad, gamma_grad, hx_grad = megalodon_ops.ema_parameters_bwd(weight_grad, bias_grad, p, log_q, gamma, hx, vander)
        return p_grad, q_grad, gamma_grad, hx_grad, None


ema_parameters = EMAParametersFunc.apply
