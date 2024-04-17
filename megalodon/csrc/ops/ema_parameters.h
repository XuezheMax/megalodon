#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>

#include "utils.h"

namespace megalodon {
namespace ops {

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                 const torch::Tensor& gamma,
                 const c10::optional<torch::Tensor>& h, int64_t L);

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersCPUFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& h, int64_t L);

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersCUDAFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& h, int64_t L);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersBwd(const torch::Tensor& w_grad,
                 const c10::optional<torch::Tensor>& b_grad,
                 const torch::Tensor& p, const torch::Tensor& log_q,
                 const torch::Tensor& gamma,
                 const c10::optional<torch::Tensor>& h,
                 const c10::optional<torch::Tensor>& v);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersCPUBwd(const torch::Tensor& w_grad,
                    const c10::optional<torch::Tensor>& b_grad,
                    const torch::Tensor& p, const torch::Tensor& log_q,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& h,
                    const c10::optional<torch::Tensor>& v);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersCUDABwd(const torch::Tensor& w_grad,
                     const c10::optional<torch::Tensor>& b_grad,
                     const torch::Tensor& p, const torch::Tensor& log_q,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& h,
                     const c10::optional<torch::Tensor>& v);

void DefineEMAParametersOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
