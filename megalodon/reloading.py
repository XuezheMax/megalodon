import os
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn.functional as F

from megalodon.config import ModelConf, TokenizerConf, logger
from megalodon.data.tokenizer import Tokenizer
from megalodon.model.mega import (
    build_model,
    Mega,
)
from megalodon.distributed import (
    init_signal_handler,
    init_torch_distributed,
    get_data_parallel_world_size,
    get_model_parallel_world_size,
    get_chunk_parallel_world_size,
    get_data_parallel_rank,
    get_model_parallel_rank,
    initialize_model_parallel,
)
from megalodon.utils import (
    get_default_half,
    get_parallel_ranks,
    get_torch_dtype
)


@dataclass
class ReloadedConf:
    world_size: int
    dp_world_size: int
    mp_world_size: int
    cp_world_size: int
    dtype: str
    is_fsdp: bool
    cfg: Dict[str, Any]

    def new_mp_world_size(self, model_parallel_size: Optional[int]):
        return (
            model_parallel_size
            if model_parallel_size is not None
            else self.mp_world_size
        )

    def new_cp_world_size(self, chunk_parallel_size: Optional[int]):
        return (
            chunk_parallel_size
            if chunk_parallel_size is not None
            else self.cp_world_size
        )

    def __post_init__(self):
        if self.is_fsdp:
            assert self.dtype == "fp32"
        else:
            assert self.mp_world_size == self.world_size // self.cp_world_size
            assert self.dp_world_size == 1


def reload_model(
    checkpoint_dir: str,
    init_distributed: bool = True,
    model_parallel_size: Optional[int] = None,
    chunk_parallel_size: Optional[int] = None,
    dtype: str = get_default_half(),
    tokenizer_path: Optional[str] = None,
) -> Tuple[Mega, Tokenizer, ReloadedConf]:
    ckpt_dir: Path = Path(checkpoint_dir)

    reloaded, tokenizer, model_cfg = reload_config_and_tokenizer(
        ckpt_dir, tokenizer_path=tokenizer_path
    )
    new_mp = reloaded.new_mp_world_size(model_parallel_size)
    new_cp = reloaded.new_cp_world_size(chunk_parallel_size)

    if init_distributed:
        init_distributed_mode(new_mp, new_cp)

    assert new_mp == get_model_parallel_world_size(), f"{new_mp} != {get_model_parallel_world_size()}"
    assert new_cp == get_chunk_parallel_world_size(), f"{new_cp} != {get_chunk_parallel_world_size()}"

    assert _is_consolidated_ckpt(ckpt_dir), "FSDP model reloading not supported."
    logger.info(
        f"Reloading consolidated model -- Path={ckpt_dir} -- "
        f"MP={get_model_parallel_world_size()}"
    )
    model = build_consolidated_model(model_cfg, dtype)
    load_consolidated_weights(ckpt_dir, model, reloaded, dtype)
    return model, tokenizer, reloaded


def model_ckpt_name(rank: int) -> str:
    return f"model.ckpt.{rank:05d}.pth"


def get_consolidated_ckpt_path(ckpt_dir: Path, mp_rank: int = 0, mp_size: int = 1):
    if mp_size == 1:
        assert mp_rank == 0
        return ckpt_dir / "consolidated.pth"
    else:
        return ckpt_dir / f"consolidated.{mp_rank:02d}.pth"


def load_consolidated_weights(ckpt_dir: Path, model: Mega, reloaded: ReloadedConf, dtype: str):
    assert not reloaded.is_fsdp
    if dtype != reloaded.dtype:
        logger.warning(
            f"Asking dtype: {dtype} when consolidated ckpt has dtype: {reloaded.dtype}"
        )

    if get_model_parallel_world_size() != reloaded.mp_world_size:
        raise ValueError(
            f"Asking model_parallel_size: {get_model_parallel_world_size()} "
            f"when checkpoint was consolidated with model_parallel_size: "
            f"{reloaded.mp_world_size}"
        )

    consolidated_ckpt_path = get_consolidated_ckpt_path(
        ckpt_dir=ckpt_dir,
        mp_rank=get_model_parallel_rank(),
        mp_size=get_model_parallel_world_size(),
    )
    logger.info("Loading consolidated ckpt...")
    state_dict = torch.load(consolidated_ckpt_path, map_location="cpu")
    logger.info("Done loading consolidated ckpt")
    load_state_dict(model, state_dict, strict=False)
    logger.info(f"Done with state-dict reloading.")


def load_state_dict(model: Mega, state_dict: Dict, strict: bool):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if len(missing_keys) > 0:
        logger.warning(f"Missing keys when reloading: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected keys when reloading: {unexpected_keys}")


def build_consolidated_model(model_cfg: ModelConf, dtype: str) -> Mega:
    logger.info(
        f"Start: building consolidated model..."
    )
    model = Mega(model_cfg)
    model.to(get_torch_dtype(dtype))
    model = model.cuda()
    logger.info(
        f"Done: building consolidated model."
    )
    return model


def init_distributed_mode(model_parallel_size: int, chunk_parallel_size: int):
    # initialize signal handler
    init_signal_handler()
    # initialize distributed mode / model parallel
    logger.info("Starting init of torch.distributed...")
    is_slurm, global_rank, world_size = init_torch_distributed()
    logger.info("Done init of torch.distributed.")

    logger.info("Starting init of model parallel...")
    initialize_model_parallel(model_parallel_size, chunk_parallel_size)
    logger.info("Done init of model parallel.")

    # print env info
    if is_slurm:
        logger.info(f"ENV: {os.environ}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"NCCL version: {torch.cuda.nccl.version()}")  # type: ignore


def reload_config_and_tokenizer(
    ckpt_dir: Path,
    tokenizer_path: Optional[str] = None,
) -> Tuple[ReloadedConf, Tokenizer, ModelConf]:
    reloaded = reload_config(ckpt_dir)
    cfg = reloaded.cfg

    tokenizer_cfg: TokenizerConf = TokenizerConf(**cfg["tokenizer"])
    old_tokenizer_path = tokenizer_cfg.path
    new_tokenizer_path: str = (
        tokenizer_path if tokenizer_path is not None else old_tokenizer_path
    )
    assert Path(new_tokenizer_path).exists(), new_tokenizer_path
    tokenizer = Tokenizer(tokenizer_cfg=tokenizer_cfg)
    model_cfg: ModelConf = ModelConf(**cfg["model"])
    model_cfg.custom_bwd = False
    model_cfg.loss_parallel = False
    model_cfg.init_mode = 'none'
    # old ckpt don't have vocab_size set
    if model_cfg.vocab_size == -1:
        model_cfg.vocab_size = tokenizer.n_words
    assert model_cfg.vocab_size == tokenizer.n_words, (
        tokenizer.n_words,
        model_cfg.vocab_size,
    )
    return reloaded, tokenizer, model_cfg


def reload_config(ckpt_dir: Path) -> ReloadedConf:
    cfg_path = ckpt_dir / "config.json"
    with cfg_path.open("r") as fp:
        cfg = json.load(fp)
    if _is_consolidated_ckpt(ckpt_dir):
        consolidate_cfg_path = ckpt_dir / "consolidate_config.json"
        if not consolidate_cfg_path.exists():
            raise RuntimeError(
                f"{consolidate_cfg_path} doesn't exists, "
                f"was the checkpoint consolidated with scripts.consolidate ?"
            )
        with consolidate_cfg_path.open("r") as fp:
            consolidate_cfg = json.load(fp)
        old_mp = consolidate_cfg["model_parallel_size"]
        old_dtype = consolidate_cfg["dtype"]
        old_ddp = 1
        old_cp = 1
        old_world_size = old_mp
        is_fsdp = False
    else:
        old_mp = cfg["model_parallel_size"]
        old_cp = cfg["chunk_parallel_size"]
        old_world_size = cfg["slurm"]["world_size"]
        assert 0 < old_mp <= old_world_size
        assert old_world_size % old_mp == 0
        old_ddp = old_world_size // old_mp // old_cp
        old_dtype = "fp32"  # FSDP training in mixed precision
        is_fsdp = True

    return ReloadedConf(
        world_size=old_world_size,
        dp_world_size=old_ddp,
        mp_world_size=old_mp,
        cp_world_size=old_cp,
        cfg=cfg,
        dtype=old_dtype,
        is_fsdp=is_fsdp,
    )


def _is_consolidated_ckpt(ckpt_dir: Path):
    return (ckpt_dir / "consolidate_config.json").exists()
