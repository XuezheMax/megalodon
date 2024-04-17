# coding=utf-8

"""Model, Chunk and data parallel groups."""

from typing import List, Optional
import logging
import torch

from .utils import ensure_divisibility

logger = logging.getLogger()

# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_MODEL_PARALLEL_RANK: Optional[int] = None
_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_DATA_PARALLEL_RANK: Optional[int] = None
_DATA_PARALLEL_WORLD_SIZE: Optional[int] = None
# Chunk parallel group that the current rank belongs to.
_CHUNK_PARALLEL_GROUP: Optional[torch.distributed.ProcessGroup] = None
_CHUNK_PARALLEL_RANK: Optional[int] = None
_CHUNK_PARALLEL_RANKS: Optional[List[int]] = None
_CHUNK_PARALLEL_WORLD_SIZE: Optional[int] = None
# rank for the prev and next chunk in the chunk parallel group
_NEXT_CHUNK_RANK: Optional[int] = None
_PREV_CHUNK_RANK: Optional[int] = None


def initialize_model_parallel(
    model_parallel_size: int,
    chunk_parallel_size: int,
    *,
    model_parallel_backend: Optional[str] = None,
    chunk_parallel_backend: Optional[str] = None,
    ddp_backend: Optional[str] = None
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    ensure_divisibility(world_size, model_parallel_size)
    ensure_divisibility(world_size, model_parallel_size * chunk_parallel_size)
    rank = torch.distributed.get_rank()

    data_parallel_size = int(world_size / (model_parallel_size * chunk_parallel_size))

    if torch.distributed.get_rank() == 0:
        logger.info("> initializing data  parallel with size {}".format(data_parallel_size))
        logger.info("> initializing chunk parallel with size {}".format(chunk_parallel_size))
        logger.info("> initializing model parallel with size {}".format(model_parallel_size))

    groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size, chunk_parallel_size, model_parallel_size)

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(chunk_parallel_size):
        for k in range(model_parallel_size):
            group = torch.distributed.new_group(groups[:, j, k].tolist(), backend=ddp_backend)
            if j == found[1] and k == found[2]:
                _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        for j in range(chunk_parallel_size):
            group = torch.distributed.new_group(groups[i, j, :].tolist(), backend=model_parallel_backend)
            if i == found[0] and j == found[1]:
                _MODEL_PARALLEL_GROUP = group

    global _CHUNK_PARALLEL_GROUP
    assert _CHUNK_PARALLEL_GROUP is None, "chunk parallel group is already initialized"
    global _CHUNK_PARALLEL_RANKS
    assert _CHUNK_PARALLEL_RANKS is None, "chunk parallel group is already initialized"
    global _PREV_CHUNK_RANK, _NEXT_CHUNK_RANK
    assert _PREV_CHUNK_RANK is None and _NEXT_CHUNK_RANK is None, "chunk parallel group is already initialized"
    for i in range(data_parallel_size):
        for k in range(model_parallel_size):
            ranks = groups[i, :, k].tolist()
            group = torch.distributed.new_group(ranks, backend=chunk_parallel_backend)
            if i == found[0] and k == found[2]:
                _CHUNK_PARALLEL_GROUP = group
                _CHUNK_PARALLEL_RANKS = ranks
                prev = found[1] - 1
                next = found[1] + 1
                _PREV_CHUNK_RANK = None if prev < 0 else ranks[prev]
                _NEXT_CHUNK_RANK = ranks[next] if next < chunk_parallel_size else None


def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None or _CHUNK_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_world_size() -> int:
    """Return world size for the model parallel group."""
    global _MODEL_PARALLEL_WORLD_SIZE
    if _MODEL_PARALLEL_WORLD_SIZE is None:
        _MODEL_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(group=get_model_parallel_group())
    return _MODEL_PARALLEL_WORLD_SIZE


def get_model_parallel_rank() -> int:
    """Return my rank for the model parallel group."""
    global _MODEL_PARALLEL_RANK
    if _MODEL_PARALLEL_RANK is None:
        _MODEL_PARALLEL_RANK = torch.distributed.get_rank(group=get_model_parallel_group())
    return _MODEL_PARALLEL_RANK


def get_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    global _DATA_PARALLEL_WORLD_SIZE
    if _DATA_PARALLEL_WORLD_SIZE is None:
        _DATA_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(group=get_data_parallel_group())
    return _DATA_PARALLEL_WORLD_SIZE


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    global _DATA_PARALLEL_RANK
    if _DATA_PARALLEL_RANK is None:
        _DATA_PARALLEL_RANK = torch.distributed.get_rank(group=get_data_parallel_group())
    return _DATA_PARALLEL_RANK


def get_chunk_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the chunk parallel group the caller rank belongs to."""
    assert _CHUNK_PARALLEL_GROUP is not None, "chunk parallel group is not initialized"
    return _CHUNK_PARALLEL_GROUP


def get_chunk_parallel_ranks() -> List[int]:
    """Get the chunk parallel group the caller rank belongs to."""
    assert _CHUNK_PARALLEL_RANKS is not None, "chunk parallel group is not initialized"
    return _CHUNK_PARALLEL_RANKS


def get_chunk_parallel_world_size() -> int:
    """Return world size for the chunk parallel group."""
    global _CHUNK_PARALLEL_WORLD_SIZE
    if _CHUNK_PARALLEL_WORLD_SIZE is None:
        _CHUNK_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(group=get_chunk_parallel_group())
    return _CHUNK_PARALLEL_WORLD_SIZE


def get_chunk_parallel_rank() -> int:
    """Return my rank for the chunk parallel group."""
    global _CHUNK_PARALLEL_RANK
    if _CHUNK_PARALLEL_RANK is None:
        _CHUNK_PARALLEL_RANK = torch.distributed.get_rank(group=get_chunk_parallel_group())
    return _CHUNK_PARALLEL_RANK


def get_chunk_parallel_prev_rank() -> Optional[int]:
    """Return the next rank for the chunk parallel group."""
    assert _CHUNK_PARALLEL_GROUP is not None, "chunk parallel group is not initialized"
    return _PREV_CHUNK_RANK


def get_chunk_parallel_next_rank() -> Optional[int]:
    """Return the next rank for the chunk parallel group."""
    assert _CHUNK_PARALLEL_GROUP is not None, "chunk parallel group is not initialized"
    return _NEXT_CHUNK_RANK


def destroy_model_parallel() -> None:
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _MODEL_PARALLEL_WORLD_SIZE
    _MODEL_PARALLEL_WORLD_SIZE = None
    global _MODEL_PARALLEL_RANK
    _MODEL_PARALLEL_RANK = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_WORLD_SIZE
    _DATA_PARALLEL_WORLD_SIZE = None
    global _DATA_PARALLEL_RANK
    _DATA_PARALLEL_RANK = None

    global _CHUNK_PARALLEL_GROUP
    _CHUNK_PARALLEL_GROUP = None
    global _CHUNK_PARALLEL_RANKS
    _CHUNK_PARALLEL_RANKS = None
    global _CHUNK_PARALLEL_WORLD_SIZE
    _CHUNK_PARALLEL_WORLD_SIZE = None
    global _CHUNK_PARALLEL_RANK
    _CHUNK_PARALLEL_RANK = None
    global _PREV_CHUNK_RANK
    _PREV_CHUNK_RANK = None
    global _NEXT_CHUNK_RANK
    _NEXT_CHUNK_RANK = None
