#include "ops/attention.h"

namespace megalodon {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor> AttentionFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, double dropout, bool use_causal_mask) {
  TORCH_CHECK(q.device().type() == torch::kCUDA);
  TORCH_CHECK(k.device().type() == torch::kCUDA);
  TORCH_CHECK(v.device().type() == torch::kCUDA);
  return AttentionCUDAFwd(q, k, v, scale, dropout, use_causal_mask);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AttentionBwd(
    const torch::Tensor& y_grad, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& w, double scale,
    bool use_causal_mask) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(q.device().type() == torch::kCUDA);
  TORCH_CHECK(k.device().type() == torch::kCUDA);
  TORCH_CHECK(v.device().type() == torch::kCUDA);
  TORCH_CHECK(w.device().type() == torch::kCUDA);
  return AttentionCUDABwd(y_grad, q, k, v, w, scale, use_causal_mask);
}

void DefineAttentionOp(py::module& m) {
  m.def("attention_fwd", &AttentionFwd, "AttentionFwd")
      .def("attention_bwd", &AttentionBwd, "AttentionBwd");
}

}  // namespace ops
}  // namespace megalodon
