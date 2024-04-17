from logging import getLogger
import torch
import torch.nn.functional as F

from megalodon.model.mega import Mega
from megalodon.utils import (
    get_torch_dtype,
    init_cache,
    truncate_cache
)
from megalodon.distributed.utils import reduce_scalar
from megalodon.modules.losses import cross_entropy

logger = getLogger()

MAX_CHUNKS_PER_BATCH = 2
TRUNCATE_CHUNKS = 8


@torch.inference_mode()
def inference(
    model: Mega,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    dtype: str,
) -> torch.Tensor:
    model.eval()

    bsz, length = tokens.shape
    compute_dtype = get_torch_dtype(dtype)

    max_length = get_max_length(length)
    tokens = F.pad(tokens, (0, max_length - length), value=0)
    targets = F.pad(targets, (0, max_length - length), value=-100)
    n_chunks = max_length // model.chunk_size
    prev_pos = n_chunks * model.chunk_size

    cache = init_cache(bsz, model, compute_dtype) if n_chunks > 0 else None
    losses = []
    for c in range(0, n_chunks, MAX_CHUNKS_PER_BATCH):
        s = c * model.chunk_size
        e = min(c + MAX_CHUNKS_PER_BATCH, n_chunks) * model.chunk_size
        # B x L x V
        x = tokens[:, s:e]
        y = targets[:, s:e]
        pred, cache = model(x, cache=cache)
        loss = cross_entropy(pred, y).view(y.size())
        losses.append(loss)

        truncate = TRUNCATE_CHUNKS * model.chunk_size <= e and (e < n_chunks * model.chunk_size or prev_pos < max_length)
        if truncate:
            ss = e - (TRUNCATE_CHUNKS - MAX_CHUNKS_PER_BATCH) * model.chunk_size
            x = tokens[:, ss:e]
            _, truncated_cache = model(x, cache=init_cache(bsz, model, compute_dtype, truncate=True))
            cache = truncate_cache(cache, truncated_cache)

    if prev_pos < max_length:
        x = tokens[:, prev_pos:]
        y = targets[:, prev_pos:]
        pred, _ = model(x, cache=cache)
        loss = cross_entropy(pred, y).view(x.size())

        losses.append(loss)

    # B x L
    loss = torch.cat(losses, dim=1)[:, :length]
    return loss


def get_max_length(length: int) -> int:
    # reduce max length prompt over all processes to have an equal number of call on each process with fsdp
    if torch.distributed.is_initialized():
        max_length = int(reduce_scalar(length, op='max'))
    else:
        max_length = length
    return max_length
