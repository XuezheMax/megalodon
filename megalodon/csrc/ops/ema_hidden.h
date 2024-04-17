#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

namespace megalodon {
namespace ops {

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h);

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenCPUFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h);

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenCUDAFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
             const torch::Tensor& p, const torch::Tensor& log_q,
             const c10::optional<torch::Tensor>& h,
             const c10::optional<torch::Tensor>& v);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenCPUBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                const torch::Tensor& p, const torch::Tensor& log_q,
                const c10::optional<torch::Tensor>& h,
                const c10::optional<torch::Tensor>& v);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                 const torch::Tensor& p, const torch::Tensor& log_q,
                 const c10::optional<torch::Tensor>& h,
                 const c10::optional<torch::Tensor>& v);

void DefineEMAHiddenOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
