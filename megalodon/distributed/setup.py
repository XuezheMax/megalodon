from typing import Tuple
import datetime
from functools import lru_cache
from logging import getLogger
import os
import sys
import signal
import socket
import subprocess
import random

import torch

logger = getLogger()


def signal_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
        sys.exit(-1)
    else:
        logger.warning("Not the master process, no need to requeue.")


def termination_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    logger.warning("Signal handler installed.")


@lru_cache()
def is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if is_torch_run():
        return int(os.environ["RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def is_master() -> bool:
    return get_global_rank() == 0


@lru_cache()
def get_master_port(job_id: int) -> int:
    if is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


@lru_cache()
def get_master_addr() -> str:
    if is_torch_run():
        return os.environ["MASTER_ADDR"]
    elif is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    else:
        return "127.0.0.1"


# ---------------- main -------------------


def init_distributed_process_group(timeout: int):
    assert isinstance(timeout, int)

    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
        timeout=datetime.timedelta(seconds=timeout),
    )


def init_torch_distributed(timeout: int = 1800) -> Tuple[bool, int, int]:
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
    """
    is_slurm = is_slurm_job()
    global_rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_ADDR"] = get_master_addr()
    os.environ["MASTER_PORT"] = str(get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1))))

    if is_torch_run():
        logger.info(f"Run launched with torchrun, local rank: {local_rank}")
    elif is_slurm_job():
        logger.info(f"Run launched with slurm, local rank: {local_rank}")
    else:
        logger.info("Single GPU job")

    # set GPU device
    torch.cuda.set_device(local_rank)

    init_distributed_process_group(timeout=timeout)  # type: ignore

    assert global_rank == torch.distributed.get_rank()
    assert world_size == torch.distributed.get_world_size()

    # sanity check
    assert 0 <= local_rank <= global_rank < world_size

    return is_slurm, global_rank, world_size
