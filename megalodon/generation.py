from logging import getLogger
from typing import List, Optional, Tuple

import torch

from megalodon.data.tokenizer import Tokenizer
from megalodon.model.mega import Mega
from megalodon.utils import (
    check_same_in_process_group,
    get_torch_dtype,
    init_cache,
    truncate_cache,
)
from megalodon.distributed.utils import reduce_scalar
from megalodon.distributed import get_model_parallel_group, get_data_parallel_world_size

logger = getLogger()

MAX_CHUNKS_PER_BATCH = 2
TRUNCATE_CHUNKS = 8


@torch.inference_mode()
def generate(
    model: Mega,
    dtype: str,
    tokenizer: Tokenizer,
    prompt: Optional[List[str]] = None,  # if None, padding example for generation
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
    truncate_chunks: int = None,
) -> List[List[int]]:
    model.eval()

    if prompt is not None:
        prompt_tokens = [tokenizer.encode(t, bos=True, eos=False) for t in prompt]
        nb_truncated_prompts = sum([max_prompt_len < len(t) for t in prompt_tokens])
    else:
        prompt_tokens, nb_truncated_prompts = None, 0

    total_truncated_prompts = reduce_scalar(nb_truncated_prompts, op='sum')
    if total_truncated_prompts > 0:
        logger.info(
            f"There are {total_truncated_prompts} prompts that are truncated on the left, "
            f"length greater than max_prompt_len = {max_prompt_len}, "
            f"maximum prompt length = {get_max_length(prompt_tokens)} across all gpus."
        )

    if prompt_tokens is not None:
        prompt_tokens = [
            t if len(t) < max_prompt_len else t[len(t) - max_prompt_len:]
            for t in prompt_tokens
        ]

    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    if prompt_tokens is None:  # padding example
        prompt_tokens = [[tokenizer.bos_id for _ in range(end_pos)]]

    bsz = len(prompt_tokens)
    if truncate_chunks is not None:
        assert bsz == 1 and get_data_parallel_world_size() == 1
        assert truncate_chunks > 0
        assert remove_prompts

    tokens = torch.full((bsz, end_pos), tokenizer.pad_id).cuda().long()
    compute_dtype = get_torch_dtype(dtype)
    cache = init_cache(bsz, model, compute_dtype)

    # copy input tokens to tensor containing generated tokens
    for k, ex_tokens in enumerate(prompt_tokens):
        tokens[k, :len(ex_tokens)] = torch.tensor(ex_tokens).long()
    prompt_mask = tokens != tokenizer.pad_id

    check_same_in_process_group(tokens, name="prompt_tokens", group_name='model', group=get_model_parallel_group())

    if truncate_chunks is not None and end_pos > truncate_chunks * model.chunk_size:
        remainder = end_pos % model.chunk_size
        tokens = tokens[:, remainder:]
        prompt_mask = prompt_mask[:, remainder:]
        start_pos = start_pos - remainder
        end_pos = end_pos - remainder
    else:
        remainder = 0

    n_chunks = (start_pos - 1) // model.chunk_size
    prev_pos = n_chunks * model.chunk_size
    for c in range(0, n_chunks, MAX_CHUNKS_PER_BATCH):
        s = c * model.chunk_size
        e = min(c + MAX_CHUNKS_PER_BATCH, n_chunks) * model.chunk_size
        _, cache = model(tokens[:, s:e], cache=cache)
        # truncate cache
        truncate = TRUNCATE_CHUNKS * model.chunk_size <= e
        if truncate:
            ss = e - (TRUNCATE_CHUNKS - MAX_CHUNKS_PER_BATCH) * model.chunk_size
            x = tokens[:, ss:e]
            _, truncated_cache = model(x, cache=init_cache(bsz, model, compute_dtype, truncate=True))
            cache = truncate_cache(cache, truncated_cache)

    logger.debug(f"prev_pos={prev_pos}, start_pos={start_pos}, end_pos={end_pos}")
    for curr_pos in range(start_pos, end_pos):
        logits, cache = model(tokens[:, prev_pos:curr_pos], cache=cache)
        # bsz x vocab
        logits = logits[:, -1]

        if use_sampling:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, top_k)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.reshape(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = torch.where(prompt_mask[:, curr_pos], tokens[:, curr_pos], next_token)
        tokens[:, curr_pos] = next_token

        prev_pos = curr_pos

    if remove_prompts:
        generated_tokens = [
            t[len(prompt_tokens[i]) - remainder:len(prompt_tokens[i]) - remainder + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    else:
        generated_tokens = [
            t[:len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    return generated_tokens


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # bsz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def get_min_length(input_tokens: Optional[List[List[int]]]) -> int:
    # reduce min length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        min_length = int(1e9)
    else:
        min_length = min([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        min_length = int(reduce_scalar(min_length, op='min'))
    return min_length


def get_max_length(input_tokens: Optional[List[List[int]]]) -> int:
    # reduce max length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        max_length = 0
    else:
        max_length = max([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        max_length = int(reduce_scalar(max_length, op='max'))
    return max_length


def get_generation_range(prompt_tokens: Optional[List[List[int]]], max_gen_len: int) -> Tuple[int, int]:
    batch_min_prompt_length = get_min_length(prompt_tokens)
    batch_max_prompt_length = get_max_length(prompt_tokens)
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len
