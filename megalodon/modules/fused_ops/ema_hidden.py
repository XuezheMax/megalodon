from typing import Optional, Tuple
import torch
from torch.autograd.function import FunctionCtx

import megalodon_extension.ops as megalodon_ops


class EMAHiddenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        hx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        log_q = q.log()
        h, v = megalodon_ops.ema_hidden_fwd(x, p, log_q, hx)
        ctx.save_for_backward(x, p, log_q, hx, v)
        return h

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        h_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, p, log_q, hx, v = ctx.saved_tensors
        x_grad, p_grad, q_grad, hx_grad = megalodon_ops.ema_hidden_bwd(h_grad, x, p, log_q, hx, v)
        return x_grad, p_grad, q_grad, hx_grad


ema_hidden = EMAHiddenFunc.apply
