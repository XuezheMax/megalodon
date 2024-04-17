from typing import Tuple

import torch
import torch.nn as nn

from .fused_ops import attention_softmax


class AttentionSoftmax(nn.Module):

    def __init__(self, dropout: float = 0.0, use_causal_mask: bool = False) -> None:
        super().__init__()
        self._dropout = dropout
        self._use_causal_mask = use_causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return attention_softmax(x, self._dropout, self._use_causal_mask, self.training)

    def extra_repr(self) -> str:
        return 'causal={}, dropout={}'.format(self._causal, self._dropout)
