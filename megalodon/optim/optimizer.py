from logging import getLogger

import torch
from torch import nn
from torch.optim import AdamW

from megalodon.config import OptimConf
from .lr_scheduler import build_lr_scheduler


logger = getLogger()


@torch.no_grad()
def rescale_grads(model: nn.Module, scale: float):
    if scale == 1.0:
        return
    p_with_grads = (p for p in model.parameters() if p.grad is not None)
    for p in p_with_grads:
        assert p.grad is not None
        p.grad /= scale


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str):
    logger.info("Starting build of optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.epsilon,
    )
    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)
    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
