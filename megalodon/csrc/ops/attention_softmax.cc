#include "ops/attention_softmax.h"

namespace megalodon {
namespace ops {

torch::Tensor AttentionSoftmaxFwd(const torch::Tensor& x, double dropout,
                                  bool use_causal_mask) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  return AttentionSoftmaxCUDAFwd(x, dropout, use_causal_mask);
}

torch::Tensor AttentionSoftmaxBwd(const torch::Tensor& y_grad,
                                  const torch::Tensor& y, bool causal_mask) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(y.device().type() == torch::kCUDA);
  return AttentionSoftmaxCUDABwd(y_grad, y, causal_mask);
}

void DefineAttentionSoftmaxOp(py::module& m) {
  m.def("attention_softmax_fwd", &AttentionSoftmaxFwd, "AttentionSoftmaxFwd")
      .def("attention_softmax_bwd", &AttentionSoftmaxBwd,
           "AttentionSoftmaxBwd");
}

}  // namespace ops
}  // namespace megalodon
