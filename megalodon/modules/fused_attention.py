import torch
import torch.nn as nn

from .fused_ops import swift_efficient_attention


class SwiftAttention(nn.Module):

    def __init__(self, scale: float = 1.0, dropout: float = 0.0, use_causal_mask: bool = True) -> None:
        super().__init__()
        self._scale = scale
        self._dropout = dropout
        self._use_causal_mask = use_causal_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return swift_efficient_attention(q, k, v, self._scale, self._dropout,
                                         self._use_causal_mask, self.training)

    def extra_repr(self) -> str:
        return 'causal={}, scale={}, dropout={}'.format(self._causal, self._scale, self._dropout)
