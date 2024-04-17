from typing import Tuple, Optional

import torch
from torch.autograd.function import FunctionCtx


class MemoryEfficientDropoutFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        p: float = 0.0,
        training: bool = True
    ) -> torch.Tensor:

        y, rng_state = memory_efficient_dropout_fwd(x, p, training)
        ctx.rng_state = rng_state
        ctx.p = p
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:

        rng_state = ctx.rng_state
        p = ctx.p
        x_grad = memory_efficient_dropout_bwd(y_grad, p, rng_state)
        return x_grad, None, None


memory_efficient_dropout = MemoryEfficientDropoutFunc.apply


def memory_efficient_dropout_fwd(
    x: torch.Tensor,
    p: float,
    training: bool,
    rng_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))

    if not training or p == 0:
        return x, None

    if rng_state is None:
        rng_state = torch.cuda.get_rng_state()
        org_rng_state = None
    else:
        # set rng state to the specific state
        org_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state)

    noise = torch.empty_like(x).bernoulli_(1 - p).div_(1 - p)
    y = x * noise

    if org_rng_state is None:
        return y, rng_state
    else:
        # reset the original rng state
        torch.cuda.set_rng_state(org_rng_state)
        return y, noise


def memory_efficient_dropout_bwd(
    y_grad: torch.Tensor,
    p: float,
    rng_state: Optional[torch.Tensor],
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:

    if p == 0:
        assert rng_state is None
        return y_grad

    # short-cut when noise mask has been provided.
    if noise is not None:
        x_grad = y_grad.mul(noise)
        return x_grad

    # set rng state to the state in fwd
    org_rng_state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng_state)

    noise = torch.empty_like(y_grad)
    noise.bernoulli_(1 - p).div_(1 - p)
    x_grad = y_grad.mul(noise)

    # reset the original rng state
    torch.cuda.set_rng_state(org_rng_state)

    return x_grad
