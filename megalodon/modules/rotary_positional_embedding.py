import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, max_positions, base=None):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.base = 10000 if base is None else base
        self.register_buffer("freqs", self._precompute_freqs())
        self.freqs_cis: Optional[torch.Tensor] = None

    def _precompute_freqs(self):
        freqs = [self.base ** (j / self.embed_dim) for j in range(0, self.embed_dim, 2)]
        freqs = torch.tensor(freqs, dtype=torch.float32)
        freqs = 1.0 / freqs
        return freqs

    @torch.no_grad()
    def _precompute_until(self, max_positions: int):
        assert self.max_positions <= max_positions
        self.max_positions = max_positions
        # C
        t = torch.arange(max_positions, dtype=torch.float32, device=self.freqs.device)
        # C x D/2
        freqs = torch.outer(t, self.freqs.float())
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def get_freqs_cis(self, start: int, end: int) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_until(self.max_positions)
        if end > self.freqs_cis.shape[0]:
            warnings.warn('Extending rotary range from {} to {}'.format(self.max_positions, end))
            self.freqs_cis = self._precompute_until(end)
        return self.freqs_cis[start:end]  # type: ignore

    def forward(self, xq, xk, start: int):
        # B x C x N x  D
        seq_len = xq.shape[1]
        freqs_cis = self.get_freqs_cis(start, start + seq_len)
        return apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, backward=False)

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}, base={:.1f}'.format(self.embed_dim, self.max_positions, self.base)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    backward: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If backward=False:
        - inputs: (xq, xk)
        - outputs: (xq_out, xk_out)
    If backward=True:
        - inputs: (grad_xq_out, grad_xk_out)
        - outputs: (grad_xq, grad_xk)
    """
    if backward:
        freqs_cis = freqs_cis.conj()

    # B x C x N x D/2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # C x 1 x D/2
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
