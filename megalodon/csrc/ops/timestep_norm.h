#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace megalodon {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
                const torch::Tensor& gamma, const torch::Tensor& beta,
                const c10::optional<torch::Tensor>& padding_mask, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCPUFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                   const torch::Tensor& prev_mean,
                   const torch::Tensor& prev_var, const torch::Tensor& gamma,
                   const torch::Tensor& beta,
                   const c10::optional<torch::Tensor>& padding_mask,
                   double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                    const torch::Tensor& prev_mean,
                    const torch::Tensor& prev_var, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormBwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                const torch::Tensor& var_grad, const torch::Tensor& X,
                const torch::Tensor& prev_mean, const torch::Tensor& count,
                const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                const torch::Tensor& gamma,
                const c10::optional<torch::Tensor>& padding_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCPUBwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                   const torch::Tensor& var_grad, const torch::Tensor& X,
                   const torch::Tensor& prev_mean, const torch::Tensor& count,
                   const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                   const torch::Tensor& gamma,
                   const c10::optional<torch::Tensor>& padding_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCUDABwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                    const torch::Tensor& var_grad, const torch::Tensor& X,
                    const torch::Tensor& prev_mean, const torch::Tensor& count,
                    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& padding_mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                     const torch::Tensor& prev_mean,
                     const torch::Tensor& prev_var, const torch::Tensor& gamma,
                     const torch::Tensor& beta,
                     const c10::optional<torch::Tensor>& padding_mask,
                     int64_t num_groups, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormCPUFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                        const torch::Tensor& prev_mean,
                        const torch::Tensor& prev_var,
                        const torch::Tensor& gamma, const torch::Tensor& beta,
                        const c10::optional<torch::Tensor>& padding_mask,
                        int64_t num_groups, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormCUDAFwd(const torch::Tensor& X,
                         const torch::Tensor& prev_count,
                         const torch::Tensor& prev_mean,
                         const torch::Tensor& prev_var,
                         const torch::Tensor& gamma, const torch::Tensor& beta,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormBwd(const torch::Tensor& Y_grad,
                     const torch::Tensor& mean_grad,
                     const torch::Tensor& var_grad, const torch::Tensor& X,
                     const torch::Tensor& prev_mean, const torch::Tensor& count,
                     const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& padding_mask,
                     int64_t num_groups);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormCPUBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups);

void DefineTimestepNormOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
