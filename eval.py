import json
import math
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, List
from pathlib import Path
import os
import fire
import torch

from megalodon.config import ValidConf
from megalodon.logger import initialize_logger
from megalodon.reloading import reload_model
from megalodon.utils import setup_env, log_host, get_default_half, set_random_seed, parse_bool_flag
from megalodon.distributed.setup import get_global_rank
from megalodon.distributed.utils import reduce_scalar
from megalodon.distributed import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_model_parallel_rank,
    get_chunk_parallel_rank,
    get_chunk_parallel_world_size
)
from megalodon.data.tokenizer import Tokenizer
from megalodon.data.dataloader import DataLoader
from megalodon.model.mega import Mega
from megalodon.generation import generate
from megalodon.inference import inference

logger = getLogger()


@dataclass
class EvalConf:
    valid: ValidConf
    checkpoint_dir: str
    dump_dir: str  # main directory where evals are dumped
    tokenizer_path: Optional[str] = None
    master_logging_only: bool = True
    dtype: str = get_default_half()
    model_parallel_size: Optional[int] = None
    seed: int = 42

    def __post_init__(self):
        checkpoint_dir = Path(self.checkpoint_dir)
        assert checkpoint_dir.is_dir(), checkpoint_dir
        if os.path.exists(self.dump_dir):
            assert os.path.isdir(self.dump_dir)
        else:
            os.makedirs(self.dump_dir, exist_ok=True)
        assert os.path.isdir(self.dump_dir)


@torch.no_grad()
def eval_ppl(
    model: Mega,
    dtype: str,
    path: str,
    tokenizer: Tokenizer,
    batch_size: int,
    world_rank: int,
    world_size: int,
):
    model.eval()

    chunk_parallel_size = get_chunk_parallel_world_size()
    assert chunk_parallel_size == 1, f"task evaluation does not support chunk parallel: {chunk_parallel_size} > 1"

    dataloader = DataLoader(
        tokenizer=tokenizer,
        path=path,
        batch_size=batch_size,
        world_rank=world_rank,
        world_size=world_size,
    )

    batches = list(iter(dataloader))
    dataloader.close()

    dummy_start = len(batches)
    len_data = reduce_scalar(len(batches), op='max')
    extra = int(len_data) - len(batches)
    if extra > 0:
        batches.extend([batches[0] for _ in range(extra)])

    loss = 0.0
    ntokens = 0
    for i, batch in enumerate(batches):
        x = batch['x'].cuda()
        y = batch['y'].cuda()
        nll = inference(model, x, y, dtype)
        if i < dummy_start:
            loss += nll.sum().item()
            ntokens += (y != -100).sum().item()

    tot_loss = reduce_scalar(loss, op='sum', group=get_data_parallel_group())
    tot_toks = reduce_scalar(ntokens, op='sum', group=get_data_parallel_group())
    return math.exp(tot_loss / tot_toks), tot_toks


@torch.no_grad()
def run_generation(
    model: Mega,
    dtype: str,
    tokenizer: Tokenizer,
    cfg: ValidConf,
    dump_dir: str,
    world_rank: int,
    world_size: int,
):
    chunk_parallel_size = get_chunk_parallel_world_size()
    assert chunk_parallel_size == 1, f"generation does not support chunk parallel: {chunk_parallel_size} > 1"

    # reload prompts
    logger.info(f"Evaluating generation on {cfg.prompt_path} ...")
    with open(cfg.prompt_path, "r") as f:
        lines = [json.loads(line.rstrip()) for line in f]
    logger.info(f"Reloaded {len(lines)} lines from {cfg.prompt_path}")

    FAKE_PROMPT = "FAKE"

    # pad / assign prompts to workers
    n_prompts = len(lines)
    n_per_worker = math.ceil(n_prompts / world_size)
    n_pad = n_per_worker * world_size - n_prompts
    prompts = [x["text"] for x in lines] + n_pad * [FAKE_PROMPT]
    a = n_per_worker * world_rank
    b = n_per_worker * (world_rank + 1)
    prompts = prompts[a:b]

    if get_model_parallel_rank() == 0 and get_chunk_parallel_rank() == 0:
        cur_save_dir = Path(dump_dir) / "generations"
        cur_save_dir.mkdir(parents=True, exist_ok=True)
        cur_save_path = cur_save_dir / f"{world_rank}.jsonl"
        f = open(cur_save_path, "a")
    else:
        cur_save_path = None
        f = None

    for i, prompt in enumerate(prompts):
        res: List[List[int]] = generate(
            model,
            dtype,
            tokenizer,
            prompt=[prompt],
            use_sampling=cfg.use_sampling,
            temp=cfg.temperature,
            max_gen_len=cfg.gen_seq_len,
            remove_prompts=True,
        )
        assert len(res) == 1, len(res)
        output = tokenizer.decode(res[0], cut_at_eos=True)
        if i < n_prompts and cur_save_path is not None:
            f.write(json.dumps({"prompt": prompt, "generation": output}, ensure_ascii=False) + "\n")

    f.close()


def main(
    checkpoint_dir: str,
    dump_dir: str,
    tokenizer_path: Optional[str] = None,
    master_logging_only: str = "true",
    dtype: str = get_default_half(),
    model_parallel_size: Optional[str] = None,
    ppl_files_str: str = "",
    prompt_path: str = "",
    batch_size: int = 1,
    use_sampling: str = "false",
    temperature: str = "1.0",
    top_k: str = "0",
    top_p: str = "0.0",
    gen_seq_len: str = "512",
    seed: str = "42"
):

    valid_cfg = ValidConf(
        ppl_files_str=ppl_files_str,
        prompt_path=prompt_path,
        batch_size=batch_size,
        use_sampling=parse_bool_flag(use_sampling),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        gen_seq_len=int(gen_seq_len),
    )
    eval_cfg = EvalConf(
        valid=valid_cfg,
        checkpoint_dir=checkpoint_dir,
        dump_dir=dump_dir,
        tokenizer_path=tokenizer_path,
        master_logging_only=parse_bool_flag(master_logging_only),
        dtype=dtype,
        model_parallel_size=int(model_parallel_size),
        seed=int(seed)
    )

    initialize_logger()
    setup_env()
    log_host()

    if get_global_rank() > 0 and eval_cfg.master_logging_only:
        logger.info(f"No print for worker {get_global_rank()}")
        logger.disabled = True

    Path(eval_cfg.dump_dir).mkdir(exist_ok=True, parents=True)

    model, tokenizer, ckpt_cfg = reload_model(
        checkpoint_dir=eval_cfg.checkpoint_dir,
        model_parallel_size=eval_cfg.model_parallel_size,
        chunk_parallel_size=1,
        dtype=eval_cfg.dtype,
        tokenizer_path=eval_cfg.tokenizer_path,
    )

    logger.info(f"checkpoint config: {ckpt_cfg}")
    logger.info(model)

    world_rank = get_data_parallel_rank()
    world_size = get_data_parallel_world_size()
    if eval_cfg.seed is not None:
        torch_seed = eval_cfg.seed + int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) + world_rank
        logger.info(f"Setting torch seed to {torch_seed}")
        set_random_seed(torch_seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = eval_cfg.dtype
    # perplexity evaluations
    logger.info("Running PPL evaluations ...")
    for path in eval_cfg.valid.ppl_files:
        logger.info(f"Evaluating PPL on {path} ...")
        ppl, n_tokens = eval_ppl(
            model=model,
            dtype=dtype,
            path=path,
            tokenizer=tokenizer,
            batch_size=eval_cfg.valid.batch_size,
            world_rank=world_rank,
            world_size=world_size
        )
        logger.info(f"PPL on {path}: {ppl}, w. {n_tokens} tokens.")
        torch.cuda.empty_cache()

    # generation
    if eval_cfg.valid.prompt_path:
        run_generation(
            model=model,
            dtype=dtype,
            tokenizer=tokenizer,
            cfg=eval_cfg.valid,
            dump_dir=eval_cfg.dump_dir,
            world_rank=world_rank,
            world_size=world_size,
        )

    logger.info("===== Finished all evaluations.")


if __name__ == "__main__":
    fire.Fire(main)
