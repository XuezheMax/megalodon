#pragma once

#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace megalodon {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> AttentionFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, double dropout, bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor> AttentionCUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, double dropout, bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AttentionBwd(
    const torch::Tensor& grad_y, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& w, double scale,
    bool use_causal_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AttentionCUDABwd(
    const torch::Tensor& grad_y, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& w, double scale,
    bool use_causal_mask);

void DefineAttentionOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
