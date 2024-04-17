import os
from typing import Union, Tuple, List, Callable, Optional
import socket
from functools import lru_cache
from logging import getLogger
from functools import partial
import contextlib
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.distributed as dist

from megalodon.distributed import FullyShardedDataParallel
from megalodon.distributed import (
    get_model_parallel_world_size,
    get_chunk_parallel_world_size,
)

Scalar = Union[int, float]
logger = getLogger()


def parse_bool_flag(arg: str) -> bool:
    if arg.lower() == 'false':
        return False
    elif arg.lower() == 'true':
         return True
    else:
        raise ValueError("Invalid value for a boolean flag!")


def get_parallel_ranks(
    global_rank: int,
    model_parallel_size: Optional[int] = None,
    chunk_parallel_size: Optional[int] = None
) -> Tuple[int, int, int]:
    if model_parallel_size is None and chunk_parallel_size is None:
        model_parallel_size = get_model_parallel_world_size()
        chunk_parallel_size = get_chunk_parallel_world_size()
    # model parallel rank
    model_parallel_rank = global_rank % model_parallel_size
    # chunk parallel rank
    global_rank = global_rank // model_parallel_size
    chunk_parallel_rank = global_rank % chunk_parallel_size
    # data parallel rank
    data_parallel_rank = global_rank // chunk_parallel_size

    return data_parallel_rank, chunk_parallel_rank, model_parallel_rank


def get_torch_dtype(dtype: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]


@lru_cache()
def get_default_half() -> str:
    return "bf16"


def get_init_fn(mode, dim=None) -> Callable[[torch.Tensor], torch.Tensor]:
    if mode == 'none':
        return lambda x: x
    elif mode == 'bert':
        std = 0.02
        init_fn = partial(nn.init.normal_, mean=0.0, std=std)
    elif mode == 'he':
        a = math.sqrt(5.0)
        init_fn = partial(nn.init.kaiming_normal_, a=a)
    elif mode == 'gaussian':
        std = 1.0 if dim is None else 1.0 / math.sqrt(dim)
        a = -3 * std
        b = 3 * std
        init_fn = partial(nn.init.trunc_normal_, mean=0.0, std=std, a=a, b=b)
    elif mode == 'xavier':
        init_fn = partial(nn.init.xavier_uniform_)
    else:
        raise ValueError('Unknown init mode: {}'.format(mode))

    return init_fn


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@contextlib.contextmanager
def create_on_gpu():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
    try:
        yield
    finally:
        torch.set_default_tensor_type(torch.FloatTensor)


@torch.no_grad()
def check_same_in_process_group(x: torch.Tensor, name: str, group_name: str, group: Optional[dist.ProcessGroup] = None):
    assert isinstance(x, torch.Tensor)
    assert x.dtype in {torch.int64, torch.int32, torch.bool}
    x = x.cuda()
    max_x = x.clone().detach()
    dist.all_reduce(max_x, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(max_x, x):
        msg = f"ISSUE: different tensor {name} detected in the same {group_name} parallel group !!! - "
        raise RuntimeError(msg)


@torch.no_grad()
def clip_grad_norm_(
    fsdp_module: FullyShardedDataParallel,
    max_norm: Union[float, int],
    norm_type: Union[float, int] = 2.0,
) -> torch.Tensor:
    """
    DUPLICATED FROM FAIRSCALE, ADDING THE REDUCTION FOR MODEL PARALLEL

    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.

    Args:
        fsdp_module (FullyShardedDataParallel): FSDP model
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
            for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    return fsdp_module.clip_grad_norm_(max_norm, norm_type)


def pad(
    tokens: List[int],
    max_length: int,
    value: int,
    padding: str = "post",
    truncating: str = "post",
):
    if len(tokens) < max_length:
        if padding == "post":
            tokens = tokens + [value] * (max_length - len(tokens))
        elif padding == "pre":
            tokens = [value] * (max_length - len(tokens)) + tokens
    if truncating == "post":
        tokens = tokens[:max_length]
    elif truncating == "pre":
        if len(tokens) > max_length:
            tokens = tokens[len(tokens) - max_length:]
    return tokens


def setup_env():
    env_vars = {
        "NCCL_DEBUG": "INFO",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_TIMEOUT": "22",
    }
    if "LOCAL_RANK" in os.environ:  # if job is started with torchrun
        env_vars.pop("NCCL_DEBUG")
    for name, value in env_vars.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
            logger.warning(f"WARNING: Setting {name} to {value}")


def log_host():
    logger.warning(f"Host: {socket.gethostname()}")
    logger.warning(f"Job hosts: {os.environ.get('SLURM_JOB_NODELIST', '')}")
    logger.warning(f"Slurm job id: {int(os.environ.get('SLURM_JOB_ID', -1))}")


def init_cache(bsz, model, compute_dtype, truncate=False):
    cache_layers = [
        (
            (
                torch.empty((bsz, 0, model.local_heads, model.z_head_dim), dtype=compute_dtype, device="cuda"),
                torch.empty((bsz, 0, model.local_heads, model.v_head_dim), dtype=compute_dtype, device="cuda"),
                0,
            ) if not truncate else None,
            (None, None, None),
            None
        )
        for _ in range(model.num_layers)
    ]
    cache_norm = (None, None, None)
    cache = (cache_layers, cache_norm)
    return cache


def truncate_cache(cache, truncated_cache):
    cache_layers, cache_output = cache
    truncated_layers, truncated_output = truncated_cache
    n_layers = len(cache_layers)
    for i in range(n_layers):
        cache_attn, cache_norm, hx = cache_layers[i]
        cache_layers[i] = (cache_attn, truncated_layers[i][1], hx)

    cache = (cache_layers, truncated_output)
    return cache
