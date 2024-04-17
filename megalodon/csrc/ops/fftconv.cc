#include "ops/fftconv.h"

namespace megalodon {
namespace ops {

torch::Tensor RFFT(const torch::Tensor& x, bool flip) {
  return RFFTCUDA(x, flip);
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvFwd(const torch::Tensor& x,
                                                    const torch::Tensor& k_f) {
  return FFTConvCUDAFwd(x, k_f);
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvBwd(
    const torch::Tensor& y_grad, const torch::Tensor& x_f,
    const torch::Tensor& k_f, const torch::Dtype& k_dtype) {
  return FFTConvCUDABwd(y_grad, x_f, k_f, k_dtype);
}

void DefineFFTConvOp(py::module& m) {
  m.def("rfft", &RFFT, "RFFT")
      .def("fftconv_fwd", &FFTConvFwd, "FFTConvFwd")
      .def(
          "fftconv_bwd",
          [](const torch::Tensor& y_grad, const torch::Tensor& x_f,
             const torch::Tensor& k_f, const py::object& k_dtype) {
            return FFTConvBwd(
                y_grad, x_f, k_f,
                torch::python::detail::py_object_to_dtype(k_dtype));
          },
          "FFTConvBwd");
}

}  // namespace ops
}  // namespace megalodon
