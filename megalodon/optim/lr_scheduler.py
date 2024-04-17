from functools import partial
import math
from torch.optim import lr_scheduler

from megalodon.config import OptimConf


def lr_linear(step: int, warmup: int, total_steps: int, init_ratio: float, end_ratio: float) -> float:
    if step < warmup:
        lr = init_ratio + float(step) * (1.0 - init_ratio) / warmup
    elif step <= total_steps:
        s = float(step - warmup) / (total_steps - warmup)
        lr = s * end_ratio + (1 - s)
    else:
        lr = end_ratio
    return lr


def lr_cosine(
    step: int,
    warmup: int,
    total_steps: int,
    cycles: float,
    init_ratio: float,
    end_ratio: float
) -> float:
    if step < warmup:
        lr = init_ratio + float(step) * (1.0 - init_ratio) / warmup
    elif step <= total_steps:
        s = float(step - warmup) / (total_steps - warmup)
        lr = end_ratio + 0.5 * (1 - end_ratio) * (math.cos(math.pi * s * cycles) + 1)
    else:
        lr = end_ratio
    return lr


def lr_constant(step: int, warmup: int, init_ratio: float) -> float:
    if step < warmup:
        return init_ratio + float(step) * (1.0 - init_ratio) / warmup
    return 1.0


def build_lr_fn(cfg: OptimConf, total_steps: int):
    if cfg.scheduler == "linear":
        lr_fn = partial(
            lr_linear,
            warmup=cfg.warmup,
            total_steps=total_steps,
            init_ratio=cfg.lr_init_ratio,
            end_ratio=cfg.lr_end_ratio
        )
    elif cfg.scheduler == "constant":
        lr_fn = partial(
            lr_constant,
            warmup=cfg.warmup,
            init_ratio=cfg.lr_init_ratio
        )
    elif cfg.scheduler == "cosine":
        lr_fn = partial(
            lr_cosine,
            warmup=cfg.warmup,
            total_steps=total_steps,
            cycles=cfg.cycles,
            init_ratio=cfg.lr_init_ratio,
            end_ratio=cfg.lr_end_ratio,
        )
    else:
        raise NotImplementedError(f"Unknown scheduler: {cfg.scheduler}")
    return lr_fn


def build_lr_scheduler(optimizer, cfg: OptimConf, total_steps: int):
    lr_fn = build_lr_fn(cfg, total_steps)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)
    return scheduler
