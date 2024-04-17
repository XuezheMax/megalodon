from typing import Optional, Union, Any
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fairscale.utils.parallel import ProcessGroupName
from fairscale.utils.params import calc_grad_norm
from fairscale.nn.data_parallel import TrainingState
from fairscale.nn import FullyShardedDataParallel as FSDP


class FullyShardedDataParallel(FSDP):
    """
        A wrapper for sharding Module parameters across data parallel workers. This
        is inspired by `Xu et al.`_ as well as the ZeRO Stage 3 from DeepSpeed_.
        FullyShardedDataParallel is commonly shorten to FSDP.

        .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
        .. _DeepSpeed: https://www.deepspeed.ai/

        Args:
            module (nn.Module):
                module to be wrapped with FSDP.
            process_group (Optional):
                process group for sharding
            process_group_reduce_scatter (Optional):
                process group for reduce scatter
                it defaults to ProcessGroupName.reduce_scatter. A seperate process group is initialized and assigned to the reduce_scatter operation. And the
                reduce_scatter operation overlaps with other operations in the backward propagation
                If it is a specific ProcessGroup, the reduce_scatter operates on this ProcessGroup, and the overlap still happens.
                To disable the overlap feature, set the process group to ProcessGroupName.default. In this case, the reduce_scatter
                operation uses the same process group with the default group.
                If reduce scatter process group size is differnt with the default process group size, the reduce_scatter
                operation rolls back to use the same process group with the default process group.
            model_parallel_process_group (Optional):
                model parallel process group for gradient clip
            chunk_parallel_process_group (Optional):
                chunk parallel process group for gradient sync
        """

    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        # The type for the process_group_reduce_scatter only can be either ProcessGroup or ProcessGroupName
        process_group_reduce_scatter: Any = ProcessGroupName.reduce_scatter,
        model_parallel_process_group: Optional[ProcessGroup] = None,
        chunk_parallel_process_group: Optional[ProcessGroup] = None,
        **kwargs
    ):
        super().__init__(
            module = module,
            process_group=process_group,
            process_group_reduce_scatter=process_group_reduce_scatter,
            **kwargs
        )
        self.model_parallel_process_group = model_parallel_process_group
        self.model_parallel_world_size = 1 if model_parallel_process_group is None else model_parallel_process_group.size()
        self.chunk_parallel_process_group = chunk_parallel_process_group
        self.chunk_parallel_world_size = 1 if chunk_parallel_process_group is None else chunk_parallel_process_group.size()
        self.chunk_parallel_buffer_size = min(int(self.bucket_cap_mb * 1024 * 1024), sum(p.numel() for p in self.parameters()))
        self.chunk_parallel_buffer = None

    def extra_repr(self) -> str:
        repr = (
            f"world_size=({self.world_size}, {self.chunk_parallel_world_size}, {self.model_parallel_world_size}), "
            f"flatten_parameters={self.flatten_parameters}, "
            f"mixed_precision={self.mixed_precision}, "
        )
        if self.verbose:
            repr = (
                f"self={id(self)} is_root={self._is_root}, "
                f"rank={self.rank}, " + repr + f"reshard_after_forward={self.reshard_after_forward}, "
                f"compute_dtype={self.compute_dtype}, "
                f"buffer_dtype={self.buffer_dtype}, "
                f"fp32_reduce_scatter={self.fp32_reduce_scatter}, "
                f"compute_device={self.compute_device}"
                f"move_params_to_cpu={self.move_params_to_cpu}, "
                f"move_grads_to_cpu={self.move_grads_to_cpu}, "
                f"bucket_cap_mb={self.bucket_cap_mb}, "
                f"clear_autocast_cache={self.clear_autocast_cache}"
                f"force_input_to_fp32={self.force_input_to_fp32}"
            )
        return repr

    @torch.no_grad()
    def clip_grad_norm_(
            self,
            max_norm: Union[float, int],
            norm_type: Union[float, int] = 2.0,
            # filter_params_fn: Callable[[Any], Any] = None,
    ) -> torch.Tensor:
        """
        Clip all gradients at this point in time. The norm is computed over all
        gradients together, as if they were concatenated into a single vector.
        Gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note:: This is analogous to `torch.nn.utils.clip_grad_norm_` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it in the OSS context would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        # We don't call torch.cuda.synchronize() here, since clipping can be
        # inside the train loop and we probably don't want to force a GPU-CPU sync.
        # _lazy_init should be sufficient, since it will force the other streams
        # to sync with the default stream (via _wait_for_previous_optim_step).
        self._lazy_init()
        assert self._is_root, "clip_grad_norm should only be called on the root (parent) instance"
        self.assert_state(TrainingState.IDLE)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        params_with_grad = self.params_with_grad
        if not self.children_share_process_group:
            raise NotImplementedError(
                "clip_grad_norm requires that all params share one process group. clip_grad_by_value_ should work"
            )
        # Computes the max norm for this shard's gradients and sync's across workers
        local_norm = calc_grad_norm(params_with_grad, norm_type).cuda()
        if norm_type == math.inf:
            total_norm = local_norm
            dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
            if self.model_parallel_world_size > 1:
                dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=self.model_parallel_process_group)
        else:
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            if self.model_parallel_world_size > 1:
                dist.all_reduce(total_norm, group=self.model_parallel_process_group)
            total_norm = total_norm ** (1.0 / norm_type)

        if self.move_grads_to_cpu:
            total_norm = total_norm.cpu()

        # Now multiply each grad by (max_norm/total_norm), same as torch 1.7 https://tinyurl.com/3wtxhhqq)
        clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-6)
        if clip_coef < 1:
            # multiply by clip_coef
            for p in params_with_grad:
                assert p.grad is not None
                p.grad.detach().mul_(clip_coef.to(p.grad.device))

        return total_norm

    def grad_all_reduce(self):
        """
        This function must be called explicitly after backward to reduce gradients.
        """

        if self.chunk_parallel_world_size == 1:
            return

        self._lazy_init()
        assert self._is_root, "grad_all_reduce should only be called on the root (parent) instance"
        self.assert_state(TrainingState.IDLE)
        if self.chunk_parallel_buffer is None:
            self.chunk_parallel_buffer = torch.empty((self.chunk_parallel_buffer_size,), dtype=torch.float32, device=torch.device("cuda"))

        params_with_grad = self.params_with_grad
        # All-reduce the gradients in buckets
        offset = 0
        buffered_params = []
        for param in params_with_grad:
            if param.grad.requires_grad:
                raise RuntimeError("DistributedDataParallel only works with gradients that don't require grad.")

            sz = param.numel()
            if sz > self.chunk_parallel_buffer_size:
                # all-reduce big params directly
                self._all_reduce_params([param])
            else:
                if offset + sz > self.chunk_parallel_buffer_size:
                    self._all_reduce_params(buffered_params)
                    offset = 0
                    buffered_params.clear()
                buffered_params.append(param)
                offset += sz

        if len(buffered_params) > 0:
            self._all_reduce_params(buffered_params)

    def _all_reduce_params(self, params):
        buffer = self.chunk_parallel_buffer
        if len(params) > 1:
            offset = 0
            for p in params:
                sz = p.numel()
                buffer[offset:offset + sz].copy_(p.grad.data.view(-1))
                offset += sz
        else:
            # we only have a single grad to all-reduce
            p = params[0]
            buffer = p.grad.data

        dist.all_reduce(buffer, group=self.chunk_parallel_process_group)

        # copy all-reduced grads back into their original place
        if len(params) > 1:
            offset = 0
            for p in params:
                sz = p.numel()
                p.grad.data.copy_(buffer[offset:offset + sz].view_as(p))
                offset += sz
