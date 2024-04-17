#include "ops/ema_hidden.h"

#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>

#include <cstdint>

namespace megalodon {
namespace ops {

namespace {

template <typename T>
void EMAHiddenCPUFwdImpl(const torch::Tensor& x, const torch::Tensor& p,
                         const torch::Tensor& log_q, const torch::Tensor& h,
                         torch::Tensor& y) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  const int64_t L = x.size(2);

  const T* x_data = x.data_ptr<T>();
  const T_ACC* p_data = p.data_ptr<T_ACC>();
  const c10::complex<T_ACC>* log_q_data = log_q.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* y_data = y.data_ptr<c10::complex<T_ACC>>();

  at::parallel_for(0, D * N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t d = i / N;
      const T_ACC p_v = p_data[i];
      const c10::complex<T_ACC> log_q_v = log_q_data[i];
      for (int64_t batch = 0; batch < B; ++batch) {
        const T* x_ptr = x_data + (batch * D + d) * L;
        c10::complex<T_ACC> sum(T_ACC(0));
        for (int64_t j = 0; j < L; ++j) {
          const c10::complex<T_ACC> qw =
              c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - j - 1));
          sum += qw * static_cast<T_ACC>(x_ptr[j]);
        }
        y_data[batch * D * N + i] = sum * p_v;
      }
    }
  });

  if (!h.defined()) {
    return;
  }

  const c10::complex<T_ACC>* h_data = h.data_ptr<c10::complex<T_ACC>>();
  at::parallel_for(0, D * N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const c10::complex<T_ACC> qw =
          c10_complex_math::exp(log_q_data[i] * static_cast<T_ACC>(L));
      for (int64_t batch = 0; batch < B; ++batch) {
        const int64_t index = batch * D * N + i;
        y_data[index] += qw * h_data[index];
      }
    }
  });
}

template <typename T>
void EMAHiddenCPUBwdImpl(const torch::Tensor& y_grad, const torch::Tensor& x,
                         const torch::Tensor& p, const torch::Tensor& log_q,
                         const torch::Tensor& h, torch::Tensor& x_grad,
                         torch::Tensor& p_grad, torch::Tensor& q_grad,
                         c10::optional<torch::Tensor>& h_grad) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  const int64_t L = x.size(2);

  const c10::complex<T_ACC>* y_grad_data =
      y_grad.data_ptr<c10::complex<T_ACC>>();
  const T* x_data = x.data_ptr<T>();
  const T_ACC* p_data = p.data_ptr<T_ACC>();
  const c10::complex<T_ACC>* log_q_data = log_q.data_ptr<c10::complex<T_ACC>>();

  T* x_grad_data = x_grad.data_ptr<T>();
  T_ACC* p_grad_data = p_grad.data_ptr<T_ACC>();
  c10::complex<T_ACC>* q_grad_data = q_grad.data_ptr<c10::complex<T_ACC>>();

  at::parallel_for(0, B * D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t d = i % D;
      const c10::complex<T_ACC>* y_grad_ptr = y_grad_data + i * N;
      const T_ACC* p_ptr = p_data + d * N;
      const c10::complex<T_ACC>* log_q_ptr = log_q_data + d * N;
      T* x_grad_ptr = x_grad_data + i * L;
      for (int64_t j = 0; j < L; ++j) {
        T_ACC sum = T_ACC(0);
        for (int64_t k = 0; k < N; ++k) {
          const c10::complex<T_ACC> qw = c10_complex_math::exp(
              log_q_ptr[k] * static_cast<T_ACC>(L - j - 1));
          sum += (std::conj(y_grad_ptr[k]) * p_ptr[k] * qw).real();
        }
        x_grad_ptr[j] = static_cast<T>(sum);
      }
    }
  });

  at::parallel_for(0, D * N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t d = i / N;
      const T_ACC p_v = p_data[i];
      const c10::complex<T_ACC> log_q_v = log_q_data[i];
      const c10::complex<T_ACC> q_v = c10_complex_math::exp(log_q_v);
      T_ACC sum1 = T_ACC(0);
      c10::complex<T_ACC> sum2(T_ACC(0));
      for (int64_t j = 0; j < L; ++j) {
        const c10::complex<T_ACC> qw1 =
            j == L - 1 ? c10::complex<T_ACC>(T_ACC(0))
                       : c10_complex_math::exp(log_q_v *
                                               static_cast<T_ACC>(L - j - 2));
        const c10::complex<T_ACC> qw2 =
            j == L - 1 ? c10::complex<T_ACC>(T_ACC(1)) : qw1 * q_v;
        for (int64_t b = 0; b < B; ++b) {
          const T_ACC x_v = static_cast<T_ACC>(x_data[(b * D + d) * L + j]);
          const c10::complex<T_ACC> dy = std::conj(y_grad_data[b * D * N + i]);
          sum1 += (dy * x_v * qw2).real();
          sum2 += dy * x_v * p_v * qw1 * static_cast<T_ACC>(L - j - 1);
        }
      }
      p_grad_data[i] = sum1;
      q_grad_data[i] = std::conj(sum2);
    }
  });

  if (!h.defined()) {
    return;
  }

  TORCH_CHECK(h_grad.has_value());
  const c10::complex<T_ACC>* h_data = h.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* h_grad_data = h_grad->data_ptr<c10::complex<T_ACC>>();
  at::parallel_for(0, D * N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const c10::complex<T_ACC> log_q_v = log_q_data[i];
      const c10::complex<T_ACC> q_v = c10_complex_math::exp(log_q_v);
      const c10::complex<T_ACC> qw1 =
          c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - 1));
      const c10::complex<T_ACC> qw2 = qw1 * q_v;
      c10::complex<T_ACC> sum(T_ACC(0));
      for (int64_t b = 0; b < B; ++b) {
        const int64_t index = b * D * N + i;
        const c10::complex<T_ACC> dy = std::conj(y_grad_data[index]);
        sum += dy * h_data[index] * qw1 * static_cast<T_ACC>(L);
        h_grad_data[index] = std::conj(dy * qw2);
      }
      q_grad_data[i] += std::conj(sum);
    }
  });
}

}  // namespace

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenCPUFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h) {
  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor y = torch::empty(
      {B, D, N}, log_q.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "EMAHiddenCPUFwd", [&]() {
        EMAHiddenCPUFwdImpl<scalar_t>(*(x.expect_contiguous()),
                                      *(p.expect_contiguous()),
                                      *(log_q.expect_contiguous()),
                                      *(h_maybe_owned->expect_contiguous()), y);
      });

  return std::make_tuple<torch::Tensor, c10::optional<torch::Tensor>>(
      std::move(y), c10::nullopt);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h) {
  if (x.device().type() == torch::kCUDA) {
    return EMAHiddenCUDAFwd(x, p, log_q, h);
  } else {
    return EMAHiddenCPUFwd(x, p, log_q, h);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenCPUBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                const torch::Tensor& p, const torch::Tensor& log_q,
                const c10::optional<torch::Tensor>& h,
                const c10::optional<torch::Tensor>& /* v */) {
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor x_grad = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> h_grad = c10::nullopt;
  if (h.has_value()) {
    h_grad = c10::make_optional(torch::empty_like(
        *h, h->options().memory_format(at::MemoryFormat::Contiguous)));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "EMAHiddenCPUBwd", [&]() {
        EMAHiddenCPUBwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(p.expect_contiguous()), *(log_q.expect_contiguous()),
            *(h_maybe_owned->expect_contiguous()), x_grad, p_grad, q_grad,
            h_grad);
      });

  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>>(
      std::move(x_grad), std::move(p_grad), std::move(q_grad),
      std::move(h_grad));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
             const torch::Tensor& p, const torch::Tensor& log_q,
             const c10::optional<torch::Tensor>& h,
             const c10::optional<torch::Tensor>& v) {
  if (x.device().type() == torch::kCUDA) {
    return EMAHiddenCUDABwd(y_grad, x, p, log_q, h, v);
  } else {
    return EMAHiddenCPUBwd(y_grad, x, p, log_q, h, v);
  }
}

void DefineEMAHiddenOp(py::module& m) {
  m.def("ema_hidden_fwd", &EMAHiddenFwd, "EMAHiddenFwd")
      .def("ema_hidden_bwd", &EMAHiddenBwd, "EMAHiddenBwd");
}

}  // namespace ops
}  // namespace megalodon
