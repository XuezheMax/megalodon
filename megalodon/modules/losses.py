from typing import Optional
from torch import Tensor
import torch.nn.functional as F

from megalodon.distributed import (
    get_chunk_parallel_rank,
    get_chunk_parallel_world_size
)
from .chunk_parallel import gather_from_chunk_parallel_region


def cross_entropy(
    logits: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""This criterion computes the cross entropy loss between input logits and target.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        logits (Tensor) : Predicted unnormalized logits;
            Shape: :math:`(B, L1, C)`.
        target (Tensor) : Ground truth class indices or class probabilities;
            Shape: :math:`(B, L2)`.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
            Default: -100
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                B ={} & \text{batch size} \\
                L1 ={} & \text{sequence length on each GPU} \\
                L2 ={} & \text{total sequence length} \\
            \end{aligned}

    """
    bsz, seq_len, _ = logits.shape
    total_len = target.size(1)
    assert seq_len * get_chunk_parallel_world_size() == total_len
    chunk_parallel_rank = get_chunk_parallel_rank()
    start = chunk_parallel_rank * seq_len
    end = (chunk_parallel_rank + 1) * seq_len
    target = target[:, start:end]
    losses = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1), reduction="none",
                             weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)
    losses = gather_from_chunk_parallel_region(losses)
    return losses
