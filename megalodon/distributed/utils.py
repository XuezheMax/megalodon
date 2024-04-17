# coding=utf-8

from typing import List, Union, Optional

import torch
import torch.distributed as dist


Scalar = Union[int, float]
ReduceOpMap = {
    "sum": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "mean": dist.ReduceOp.AVG,
    "avg": dist.ReduceOp.AVG,
}


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_specific_dim(tensor: torch.Tensor, num_partitions: int, split_dim: int,
                                    contiguous_split_chunks: bool = False) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        split_dim: the dimension along which to split
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    split_dim_size = divide_and_check_no_remainder(tensor.shape[split_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, split_dim_size, dim=split_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return [chunk.contiguous() for chunk in tensor_list]

    return tensor_list


def reduce_scalar(x: Scalar, op: str, group: Optional[dist.ProcessGroup] = None) -> Scalar:
    tensor = torch.tensor(x).cuda()
    torch.distributed.all_reduce(tensor, op=ReduceOpMap[op.lower()], group=group)
    out = tensor.item()
    return out
