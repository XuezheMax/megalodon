from typing import Tuple, Optional
import importlib
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

import apex
fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
from apex.normalization.fused_layer_norm import fused_layer_norm_affine, fused_layer_norm
from apex.normalization.fused_layer_norm import manual_rms_norm, fused_rms_norm_affine, fused_rms_norm


class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(*normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input_):
        weight = None if self.weight is None else self.weight + 1.0
        if torch.jit.is_tracing() or torch.jit.is_scripting() or not input_.is_cuda:
            return F.layer_norm(input_, self.normalized_shape, weight, self.bias, self.eps)

        if self.elementwise_affine:
            return fused_layer_norm_affine(input_, weight, self.bias, self.normalized_shape, self.eps)
        else:
            return fused_layer_norm(input_, self.normalized_shape, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class FusedRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, input_):
        weight = None if self.weight is None else self.weight + 1.0
        if torch.jit.is_tracing() or torch.jit.is_scripting() or not input_.is_cuda:
            return manual_rms_norm(input_, self.normalized_shape, weight, self.eps)

        if self.elementwise_affine:
            return fused_rms_norm_affine(input_, weight, self.normalized_shape, self.eps)
        else:
            return fused_rms_norm(input_, self.normalized_shape, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def layernorm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    dim: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight is None:
        return fused_layer_norm_cuda.forward(x, (dim,), eps)
    else:
        weight = weight + 1.0
        return fused_layer_norm_cuda.forward_affine(x, (dim,), weight, bias, eps)


def layernorm_bwd(
    y_grad: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    invvar: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    dim: int,
    eps: float
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if weight is None:
        x_grad = fused_layer_norm_cuda.backward(y_grad, mean, invvar, x, (dim,), eps)
        return x_grad, None, None
    else:
        weight = weight + 1.0
        return fused_layer_norm_cuda.backward_affine(y_grad, mean, invvar, x, (dim,), weight, bias, eps)


def rmsnorm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    dim: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight is None:
        return fused_layer_norm_cuda.rms_forward(x, (dim,), eps)
    else:
        weight = weight + 1.0
        return fused_layer_norm_cuda.rms_forward_affine(x, (dim,), weight, eps)


def rmsnorm_bwd(
    y_grad: torch.Tensor,
    x: torch.Tensor,
    invvar: torch.Tensor,
    weight: Optional[torch.Tensor],
    dim: int,
    eps: float
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if weight is None:
        return fused_layer_norm_cuda.rms_backward(y_grad, invvar, x, (dim,), eps), None
    else:
        weight = weight + 1.0
        return fused_layer_norm_cuda.rms_backward_affine(y_grad, invvar, x, (dim,), weight, eps)
