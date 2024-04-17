#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace megalodon {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                const torch::Tensor& beta,
                const c10::optional<torch::Tensor>& padding_mask, double eps,
                bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCPUFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                   const torch::Tensor& beta,
                   const c10::optional<torch::Tensor>& padding_mask, double eps,
                   bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCPUBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GroupSequenceNormFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                     const torch::Tensor& beta,
                     const c10::optional<torch::Tensor>& padding_mask,
                     int64_t num_groups, double eps, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GroupSequenceNormCPUFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                        const torch::Tensor& beta,
                        const c10::optional<torch::Tensor>& padding_mask,
                        int64_t num_groups, double eps, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GroupSequenceNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                         const torch::Tensor& beta,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, double eps, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupSequenceNormBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GroupSequenceNormCPUBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups,
    bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GroupSequenceNormCUDABwd(const torch::Tensor& Y_grad, const torch::Tensor& X,
                         const torch::Tensor& count, const torch::Tensor& mean,
                         const torch::Tensor& rstd, const torch::Tensor& gamma,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, bool length_last);

void DefineSequenceNormOp(py::module& m);

}  // namespace ops
}  // namespace megalodon
