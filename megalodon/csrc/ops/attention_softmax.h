#pragma once

#include <torch/torch.h>

#include "utils.h"

namespace megalodon {
namespace ops {

torch::Tensor AttentionSoftmaxFwd(const torch::Tensor& x, double dropout,
                                  bool use_causal_mask);
torch::Tensor AttentionSoftmaxCUDAFwd(const torch::Tensor& x, double dropout,
                                      bool use_causal_mask);

torch::Tensor AttentionSoftmaxBwd(const torch::Tensor& y_grad,
                                  const torch::Tensor& y, bool use_causal_mask);
torch::Tensor AttentionSoftmaxCUDABwd(const torch::Tensor& y_grad,
                                      const torch::Tensor& y,
                                      bool use_causal_mask);

void DefineAttentionSoftmaxOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
