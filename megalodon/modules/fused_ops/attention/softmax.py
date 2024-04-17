from typing import Tuple

import torch
from torch.autograd.function import FunctionCtx

import megalodon_extension.ops as megalodon_ops


class AttentionSoftmaxFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        training: bool = True
    ) -> torch.Tensor:
        y = megalodon_ops.attention_softmax_fwd(x, dropout if training else 0.0, use_causal_mask)
        ctx.save_for_backward(y)
        ctx.use_causal_mask = use_causal_mask
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        y, = ctx.saved_tensors
        use_causal_mask = ctx.use_causal_mask
        x_grad = megalodon_ops.attention_softmax_bwd(y_grad, y, use_causal_mask)
        return x_grad, None, None, None


attention_softmax = AttentionSoftmaxFunc.apply
