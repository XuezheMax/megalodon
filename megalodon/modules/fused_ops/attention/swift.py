from typing import Tuple, Optional

import torch
from torch.autograd.function import FunctionCtx

import megalodon_extension.ops as megalodon_ops


class FusedAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float = 1.0,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        training: bool = True
    ) -> torch.Tensor:
        p = dropout if training else 0.0
        y, w = _fused_efficient_attention_fwd(q, k, v, scale, p, use_causal_mask)
        ctx.save_for_backward(q, k, v, w)
        ctx.scale = scale
        ctx.use_causal_mask = use_causal_mask
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        q, k, v, w = ctx.saved_tensors
        scale = ctx.scale
        use_causal_mask = ctx.use_causal_mask
        q_grad, k_grad, v_grad = _fused_efficient_attention_bwd(y_grad, q, k, v, w, scale, use_causal_mask)
        return q_grad, k_grad, v_grad, None, None, None, None


def _fused_efficient_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    dropout: float = 0.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y, w = megalodon_ops.attention_fwd(q, k, v, scale, dropout, use_causal_mask)
    return y, w


def _fused_efficient_attention_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    scale: float = 1.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_grad, k_grad, v_grad = megalodon_ops.attention_bwd(grad, q, k, v, w, scale, use_causal_mask)
    return q_grad, k_grad, v_grad


swift_efficient_attention = FusedAttentionFunc.apply


def swift_efficient_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    dropout: float = 0.0,
    use_causal_mask: bool = True,
    training: bool = True,
    requires_grad: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert training or not requires_grad
    p = dropout if training else 0.0
    y, w = _fused_efficient_attention_fwd(q, k, v, scale, p, use_causal_mask)
    w = w if requires_grad else None
    return y, w


def swift_efficient_attention_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    scale: float = 1.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_efficient_attention_bwd(grad, q, k, v, w, scale, use_causal_mask)
