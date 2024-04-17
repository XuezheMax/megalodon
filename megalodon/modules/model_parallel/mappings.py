# coding=utf-8

# Most parts of the code here are adapted from Fairscale
# repo: https://github.com/facebookresearch/fairscale

from typing import Any, Optional

import torch

from megalodon.distributed import (
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_model_parallel_rank,
)
from megalodon.distributed.utils import split_tensor_along_specific_dim


def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    group = get_model_parallel_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)

    return input_


def _split(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Split the tensor along a specific dimension and keep the corresponding slice."""
    world_size = get_model_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along the specific dimension.
    split_dim = input_.dim() - 1 if dim is None else dim
    input_list = split_tensor_along_specific_dim(input_, world_size, split_dim=split_dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Gather tensors and concatinate along the specific dimension."""
    group = get_model_parallel_group()
    world_size = get_model_parallel_world_size()

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


def _reduce_scatter(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_model_parallel_group()
    world_size = get_model_parallel_world_size()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # reduce-scatter.
    rank = get_model_parallel_rank()
    scatter_dim = input_.dim() - 1 if dim is None else dim
    input_list = split_tensor_along_specific_dim(input_, world_size, split_dim=scatter_dim, contiguous_split_chunks=True)
    output = torch.empty_like(input_list[rank])
    torch.distributed.reduce_scatter(output, list(input_list), group=group)

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _split(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _gather(grad_output, dim), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _gather(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _split(grad_output, dim), None


class _GatherCopyModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _gather(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _reduce_scatter(grad_output, dim), None


class _ReduceScatterModelParallelRegion(torch.autograd.Function):
    """Redcue-scatter the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, dim):  # type: ignore
        ctx.dim = dim
        return _reduce_scatter(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        dim = ctx.dim
        return _gather(grad_output, dim), None

# -----------------
# Helper functions.
# -----------------


def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_model_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, dim)


def gather_copy_model_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _GatherCopyModelParallelRegion.apply(input_, dim)


def reduce_scatter_model_parallel_region(input_: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return _ReduceScatterModelParallelRegion.apply(input_, dim)
