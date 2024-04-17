#pragma once

#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace megalodon {
namespace ops {

torch::Tensor RFFT(const torch::Tensor& x, bool flip);
torch::Tensor RFFTCUDA(const torch::Tensor& x, bool flip);

std::tuple<torch::Tensor, torch::Tensor> FFTConvFwd(const torch::Tensor& x,
                                                    const torch::Tensor& k_f);
std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDAFwd(
    const torch::Tensor& x, const torch::Tensor& k_f);

std::tuple<torch::Tensor, torch::Tensor> FFTConvBwd(
    const torch::Tensor& y_grad, const torch::Tensor& x_f,
    const torch::Tensor& k_f, const torch::Dtype& k_dtype);
std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x_f,
    const torch::Tensor& k_f, const torch::Dtype& k_dtype);

void DefineFFTConvOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
