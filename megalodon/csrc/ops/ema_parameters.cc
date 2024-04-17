#include "ops/ema_parameters.h"

#include <ATen/Parallel.h>
#include <asm-generic/errno.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>

#include <cstring>
#include <utility>
#include <vector>

namespace megalodon {
namespace ops {

namespace {

template <typename T>
void EMAParametersCPUFwdImpl(const torch::Tensor& p, const torch::Tensor& log_q,
                             const torch::Tensor& gamma, const torch::Tensor& h,
                             int64_t L, torch::Tensor& w,
                             c10::optional<torch::Tensor>& b) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);

  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* w_data = w.data_ptr<T>();

  std::vector<c10::complex<T>> c(D * N);
  at::parallel_for(0, D * N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      c[i] = p_data[i] * gamma_data[i];
    }
  });

  at::parallel_for(0, D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const c10::complex<T>* log_q_ptr = log_q_data + i * N;
      const c10::complex<T>* c_ptr = c.data() + i * N;
      T* w_ptr = w_data + i * L;
      for (int64_t j = 0; j < L; ++j) {
        T sum = T(0);
        for (int64_t k = 0; k < N; ++k) {
          const c10::complex<T> qw =
              c10_complex_math::exp(log_q_ptr[k] * static_cast<T>(j));
          sum += (c_ptr[k] * qw).real();
        }
        w_ptr[j] = sum;
      }
    }
  });

  if (!h.defined()) {
    return;
  }
  const int64_t B = h.size(0);
  const c10::complex<T>* h_data = h.data_ptr<c10::complex<T>>();
  T* b_data = b->data_ptr<T>();
  at::parallel_for(0, B * D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t d = i % D;
      const c10::complex<T>* log_q_ptr = log_q_data + d * N;
      const c10::complex<T>* gamma_ptr = gamma_data + d * N;
      const c10::complex<T>* h_ptr = h_data + i * N;
      T* b_ptr = b_data + i * L;
      for (int64_t j = 0; j < L; ++j) {
        T sum = T(0);
        for (int64_t k = 0; k < N; ++k) {
          const c10::complex<T> qw =
              c10_complex_math::exp(log_q_ptr[k] * static_cast<T>(j + 1));
          sum += (qw * gamma_ptr[k] * h_ptr[k]).real();
        }
        b_ptr[j] = sum;
      }
    }
  });
}

template <typename T>
std::tuple<T, c10::complex<T>, c10::complex<T>> RowwiseEMAWeightCPUBwd(
    int64_t L, const T* w_grad, T p, c10::complex<T> log_q,
    c10::complex<T> gamma) {
  const c10::complex<T> q = c10_complex_math::exp(log_q);
  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t i = 0; i < L; ++i) {
    const T dw = w_grad[i];
    const c10::complex<T> qw1 =
        i == 0 ? c10::complex<T>(T(0))
               : c10_complex_math::exp(log_q * static_cast<T>(i - 1));
    const c10::complex<T> qw2 = i == 0 ? c10::complex<T>(T(1)) : qw1 * q;
    sum1 += dw * qw1 * static_cast<T>(i);
    sum2 += dw * qw2;
  }
  return std::make_tuple((sum2 * gamma).real(), std::conj(sum1 * p * gamma),
                         std::conj(sum2 * p));
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
RowwiseEMABiasCPUBwd(int64_t L, const T* b_grad, c10::complex<T> log_q,
                     c10::complex<T> gamma, c10::complex<T> h) {
  const c10::complex<T> q = c10_complex_math::exp(log_q);
  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t i = 0; i < L; ++i) {
    const T db = b_grad[i];
    const c10::complex<T> qw1 =
        c10_complex_math::exp(log_q * static_cast<T>(i));
    const c10::complex<T> qw2 = qw1 * q;
    sum1 += db * qw1 * static_cast<T>(i + 1);
    sum2 += db * qw2;
  }
  return std::make_tuple(std::conj(sum1 * gamma * h), std::conj(sum2 * h),
                         std::conj(sum2 * gamma));
}

template <typename T>
void EMAParametersCPUBwdImpl(const torch::Tensor& w_grad,
                             const torch::Tensor& b_grad,
                             const torch::Tensor& p, const torch::Tensor& log_q,
                             const torch::Tensor& gamma, const torch::Tensor& h,
                             torch::Tensor& p_grad, torch::Tensor& q_grad,
                             torch::Tensor& gamma_grad,
                             c10::optional<torch::Tensor>& h_grad) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);
  const int64_t L = w_grad.size(-1);

  const T* w_grad_data = w_grad.data_ptr<T>();
  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* p_grad_data = p_grad.data_ptr<T>();
  c10::complex<T>* q_grad_data = q_grad.data_ptr<c10::complex<T>>();
  c10::complex<T>* gamma_grad_data = gamma_grad.data_ptr<c10::complex<T>>();

  std::vector<c10::complex<T>> q_pow(D * N, c10::complex<T>(T(1)));

  at::parallel_for(0, D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* w_grad_ptr = w_grad_data + i * L;
      const T* p_ptr = p_data + i * N;
      const c10::complex<T>* log_q_ptr = log_q_data + i * N;
      const c10::complex<T>* gamma_ptr = gamma_data + i * N;
      T* p_grad_ptr = p_grad_data + i * N;
      c10::complex<T>* q_grad_ptr = q_grad_data + i * N;
      c10::complex<T>* gamma_grad_ptr = gamma_grad_data + i * N;
      for (int64_t j = 0; j < N; ++j) {
        std::tie(p_grad_ptr[j], q_grad_ptr[j], gamma_grad_ptr[j]) =
            RowwiseEMAWeightCPUBwd(L, w_grad_ptr, p_ptr[j], log_q_ptr[j],
                                   gamma_ptr[j]);
      }
    }
  });

  if (!b_grad.defined()) {
    return;
  }

  TORCH_CHECK(h.defined());
  const int64_t B = b_grad.size(0);
  const T* b_grad_data = b_grad.data_ptr<T>();
  const c10::complex<T>* h_data = h.data_ptr<c10::complex<T>>();
  c10::complex<T>* h_grad_data = h_grad->data_ptr<c10::complex<T>>();

  at::parallel_for(0, D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const c10::complex<T>* log_q_ptr = log_q_data + i * N;
      const c10::complex<T>* gamma_ptr = gamma_data + i * N;
      c10::complex<T>* q_grad_ptr = q_grad_data + i * N;
      c10::complex<T>* gamma_grad_ptr = gamma_grad_data + i * N;
      for (int64_t batch = 0; batch < B; ++batch) {
        const T* b_grad_ptr = b_grad_data + (batch * D + i) * L;
        const c10::complex<T>* h_ptr = h_data + (batch * D + i) * N;
        c10::complex<T>* h_grad_ptr = h_grad_data + (batch * D + i) * N;
        for (int64_t j = 0; j < N; ++j) {
          auto [dq, dgamma, dh] = RowwiseEMABiasCPUBwd(
              L, b_grad_ptr, log_q_ptr[j], gamma_ptr[j], h_ptr[j]);
          q_grad_ptr[j] += dq;
          gamma_grad_ptr[j] += dgamma;
          h_grad_ptr[j] = dh;
        }
      }
    }
  });
}

}  // namespace

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersCPUFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& h, int64_t L) {
  const int64_t D = p.size(0);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor w = torch::empty(
      {D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> b = c10::nullopt;
  if (h.has_value()) {
    const int64_t B = h->size(0);
    b = c10::make_optional(torch::empty(
        {B, D, L}, p.options().memory_format(at::MemoryFormat::Contiguous)));
  }

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAParametersCPUFwd", [&]() {
    EMAParametersCPUFwdImpl<scalar_t>(
        *(p.expect_contiguous()), *(log_q.expect_contiguous()),
        *(gamma.expect_contiguous()), *(h_maybe_owned->expect_contiguous()), L,
        w, b);
  });

  return std::make_tuple<torch::Tensor, c10::optional<torch::Tensor>,
                         c10::optional<torch::Tensor>>(
      std::move(w), std::move(b), c10::nullopt);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                 const torch::Tensor& gamma,
                 const c10::optional<torch::Tensor>& h, int64_t L) {
  if (p.device().type() == torch::kCUDA) {
    return EMAParametersCUDAFwd(p, log_q, gamma, h, L);
  } else {
    return EMAParametersCPUFwd(p, log_q, gamma, h, L);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersCPUBwd(const torch::Tensor& w_grad,
                    const c10::optional<torch::Tensor>& b_grad,
                    const torch::Tensor& p, const torch::Tensor& log_q,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& h,
                    const c10::optional<torch::Tensor>& /* v */) {
  c10::MaybeOwned<torch::Tensor> b_grad_maybe_owned =
      at::borrow_from_optional_tensor(b_grad);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> h_grad = c10::nullopt;
  if (h.has_value()) {
    TORCH_CHECK(b_grad.has_value());
    h_grad = c10::make_optional(torch::empty_like(
        *h, h->options().memory_format(at::MemoryFormat::Contiguous)));
  }

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAParametersCPUBwd", [&]() {
    EMAParametersCPUBwdImpl<scalar_t>(
        *(w_grad.expect_contiguous()),
        *(b_grad_maybe_owned->expect_contiguous()), *(p.expect_contiguous()),
        *(log_q.expect_contiguous()), *(gamma.expect_contiguous()),
        *(h_maybe_owned->expect_contiguous()), p_grad, q_grad, gamma_grad,
        h_grad);
  });

  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>>(
      std::move(p_grad), std::move(q_grad), std::move(gamma_grad),
      std::move(h_grad));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersBwd(const torch::Tensor& w_grad,
                 const c10::optional<torch::Tensor>& b_grad,
                 const torch::Tensor& p, const torch::Tensor& log_q,
                 const torch::Tensor& gamma,
                 const c10::optional<torch::Tensor>& h,
                 const c10::optional<torch::Tensor>& v) {
  if (p.device().type() == torch::kCUDA) {
    return EMAParametersCUDABwd(w_grad, b_grad, p, log_q, gamma, h, v);
  } else {
    return EMAParametersCPUBwd(w_grad, b_grad, p, log_q, gamma, h, v);
  }
}

void DefineEMAParametersOp(py::module& m) {
  m.def("ema_parameters_fwd", &EMAParametersFwd, "EMAParametersFwd")
      .def("ema_parameters_bwd", &EMAParametersBwd, "EMAParametersBwd");
}

}  // namespace ops
}  // namespace megalodon
