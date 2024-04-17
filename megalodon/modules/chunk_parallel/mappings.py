# coding=utf-8

from typing import Any, Optional

import torch

from megalodon.distributed import (
    get_chunk_parallel_group,
    get_chunk_parallel_world_size,
    get_chunk_parallel_rank,
    get_chunk_parallel_prev_rank,
    get_chunk_parallel_next_rank,
)
from megalodon.distributed.utils import split_tensor_along_specific_dim


def should_send_to_next():
    return get_chunk_parallel_world_size() > 1 and get_chunk_parallel_next_rank() is not None


def should_recv_from_prev():
    return get_chunk_parallel_world_size() > 1 and get_chunk_parallel_prev_rank() is not None


def _send_tensor_to_next(tensor: torch.Tensor, async_op: bool = False):
    """send the input tensor to the next gpu in the chunk parallel group."""
    group = get_chunk_parallel_group()
    dst = get_chunk_parallel_next_rank()

    # Send to destination.
    if async_op:
        handle = torch.distributed.isend(tensor, dst, group=group)
    else:
        torch.distributed.send(tensor, dst, group=group)
        handle = None

    return tensor, handle


def _send_grad_to_prev(grad: torch.Tensor, async_op: bool = False):
    """send the input tensor to the next gpu in the chunk parallel group."""
    group = get_chunk_parallel_group()
    dst = get_chunk_parallel_prev_rank()

    # Send to destination.
    if async_op:
        handle = torch.distributed.isend(grad, dst, group=group)
    else:
        torch.distributed.send(grad, dst, group=group)
        handle = None

    return grad, handle


def _receive_grad_from_next(ctx: Any, grad: torch.Tensor, async_op: bool = False):
    """receive the input tensor from the previous gpu in the chunk parallel group."""
    group = get_chunk_parallel_group()
    src = get_chunk_parallel_next_rank()

    if ctx:
        ctx.mark_dirty(grad)

    # Recevie from source.
    # TODO: add original grad to recv grad
    if async_op:
        handle = torch.distributed.irecv(grad, src, group=group)
    else:
        torch.distributed.recv(grad, src, group=group)
        handle = None

    return grad, handle


def _receive_tensor_from_prev(ctx: Any, tensor: torch.Tensor, async_op: bool = False):
    """receive the input tensor from the previous gpu in the chunk parallel group."""
    group = get_chunk_parallel_group()
    src = get_chunk_parallel_prev_rank()

    if ctx:
        ctx.mark_dirty(tensor)

    # Recevie from source.
    if async_op:
        handle = torch.distributed.irecv(tensor, src, group=group)
    else:
        torch.distributed.recv(tensor, src, group=group)
        handle = None

    return tensor, handle


def _split(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Split the tensor along the specific dimension and keep the corresponding slice."""
    world_size = get_chunk_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along specific dimension.
    split_dim = input_.dim() - 1 if dim is None else dim
    input_list = split_tensor_along_specific_dim(input_, world_size, split_dim=split_dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_chunk_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Gather tensors and concatinate along the specific dimension."""
    group = get_chunk_parallel_group()
    world_size = get_chunk_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # shape and dimension.
    gather_dim = input_.dim() - 1 if dim is None else dim
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=gather_dim)

    return output


class _SendToNextChunkParallelRegion(torch.autograd.Function):
    """Send the input to the next chunk parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        out, handle = _send_tensor_to_next(input_)
        assert handle is None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore
        grad_inp, handle = _receive_grad_from_next(None, grad_out)
        assert handle is None
        return grad_inp


class _ReceiveFromPreviousChunkParallelRegion(torch.autograd.Function):
    """Receive the input from the previous chunk parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        out, handle = _receive_tensor_from_prev(None, input_)
        assert handle is None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore
        grad_inp, handle = _send_grad_to_prev(grad_out)
        assert handle is None
        return grad_inp


class _GatherFromChunkParallelRegion(torch.autograd.Function):
    """Gather the input from chunk parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _gather(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _split(grad_output, dim), None


# -----------------
# Helper functions.
# -----------------


def send_to_next_chunk_parallel_region(input_: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return _SendToNextChunkParallelRegion.apply(input_)


def recv_from_prev_chunk_parallel_region(input_: torch.Tensor) -> Optional[torch.Tensor]:
    return _ReceiveFromPreviousChunkParallelRegion.apply(input_)


def gather_from_chunk_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _GatherFromChunkParallelRegion.apply(input_, dim)
