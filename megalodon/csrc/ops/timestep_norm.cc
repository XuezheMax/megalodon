#include "ops/timestep_norm.h"

#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace megalodon {
namespace ops {

namespace {

template <typename T>
void TimestepNormCPUFwdImpl(
    const torch::Tensor& X, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, double eps, torch::Tensor& Y,
    torch::Tensor& count, torch::Tensor& mean, torch::Tensor& var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);

  const T* X_data = X.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T* cummean_data = cummean.data_ptr<T>();
  T* cumrstd_data = cumrstd.data_ptr<T>();

  std::vector<T_ACC> u(B * H, T_ACC(0));
  std::vector<T_ACC> v(B * H, T_ACC(0));

  at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* X_ptr = X_data + i * L * H;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + i * L;

      T* Y_ptr = Y_data + i * L * H;
      T* m1_ptr = mean_data + i * H;
      T* m2_ptr = var_data + i * H;
      T* cu_ptr = cummean_data + i * L * H;
      T* cr_ptr = cumrstd_data + i * L * H;

      T_ACC* u_ptr = u.data() + i * H;
      T_ACC* v_ptr = v.data() + i * H;

      count_data[i] = prev_count_data[i];
      int64_t m0 = prev_count_data[i];
      for (int64_t j = 0; j < H; ++j) {
        u_ptr[j] = static_cast<T_ACC>(prev_mean_data[i * H + j]);
        v_ptr[j] = static_cast<T_ACC>(prev_var_data[i * H + j]);
      }

      for (int64_t j = 0; j < L; ++j) {
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        for (int64_t k = 0; k < H; ++k) {
          const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
          const T_ACC w = static_cast<T_ACC>(gamma_data[k]);
          const T_ACC b = static_cast<T_ACC>(beta_data[k]);
          const auto [_, m1, m2] =
              utils::WelfordUpdate(m0, u_ptr[k], v_ptr[k], x);
          const T_ACC rstd = T(1) / std::sqrt(m2 + static_cast<T_ACC>(eps));
          Y_ptr[j * H + k] =
              mask ? T(0) : static_cast<T>((x - m1) * rstd * w + b);
          u_ptr[k] = mask ? u_ptr[k] : m1;
          v_ptr[k] = mask ? v_ptr[k] : m2;
          cu_ptr[j * H + k] = static_cast<T>(m1);
          cr_ptr[j * H + k] = static_cast<T>(rstd);
        }
        m0 += mask ? 0 : 1;
        count_data[i] = m0;
      }
      for (int64_t j = 0; j < H; ++j) {
        m1_ptr[j] = static_cast<T>(u_ptr[j]);
        m2_ptr[j] = static_cast<T>(v_ptr[j]);
      }
    }
  });
}

template <typename T>
void TimestepNormCPUBwdImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& padding_mask,
    torch::Tensor& X_grad, torch::Tensor& prev_mean_grad,
    torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T* cummean_data = cummean.data_ptr<T>();
  const T* cumrstd_data = cumrstd.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* m1_grad_data = prev_mean_grad.data_ptr<T>();
  T* m2_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();

  std::vector<T_ACC> u_grad(B * H, T_ACC(0));
  std::vector<T_ACC> v_grad(B * H, T_ACC(0));
  std::vector<T_ACC> w_grad(B * H, T_ACC(0));
  std::vector<T_ACC> b_grad(B * H, T_ACC(0));

  at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* Y_grad_ptr = Y_grad_data + i * L * H;
      const T* mean_grad_ptr = mean_grad_data + i * H;
      const T* var_grad_ptr = var_grad_data + i * H;
      const T* X_ptr = X_data + i * L * H;
      const T* mean_ptr = cummean_data + i * L * H;
      const T* rstd_ptr = cumrstd_data + i * L * H;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + i * L;

      T* X_grad_ptr = X_grad_data + i * L * H;
      T* m1_grad_ptr = m1_grad_data + i * H;
      T* m2_grad_ptr = m2_grad_data + i * H;

      T_ACC* u_grad_ptr = u_grad.data() + i * H;
      T_ACC* v_grad_ptr = v_grad.data() + i * H;
      T_ACC* w_grad_ptr = w_grad.data() + i * H;
      T_ACC* b_grad_ptr = b_grad.data() + i * H;

      int64_t m0 = count_data[i];
      for (int64_t j = 0; j < H; ++j) {
        u_grad_ptr[j] = static_cast<T_ACC>(mean_grad_ptr[j]);
        v_grad_ptr[j] = static_cast<T_ACC>(var_grad_ptr[j]);
      }

      for (int64_t j = L - 1; j >= 0; --j) {
        const T* prev_mean_ptr =
            j == 0 ? prev_mean_data + i * H : mean_ptr + (j - 1) * H;
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(m0);
        m0 -= mask ? 0 : 1;
        for (int64_t k = 0; k < H; ++k) {
          const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + k]);
          const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
          const T_ACC prev_u = static_cast<T_ACC>(prev_mean_ptr[k]);
          const T_ACC u = static_cast<T_ACC>(mean_ptr[j * H + k]);
          const T_ACC r = static_cast<T_ACC>(rstd_ptr[j * H + k]);
          const T_ACC w = static_cast<T_ACC>(gamma_data[k]);
          const T_ACC dy_rstd = dy * r;
          const T_ACC delta = x - u;
          const T_ACC dv =
              v_grad_ptr[k] - (T_ACC(0.5) * dy * w * delta * utils::Cube(r));
          const T_ACC du =
              u_grad_ptr[k] - (w * dy_rstd + coef * dv * (x - prev_u));
          const T_ACC dx =
              w * dy_rstd + dv * coef * (delta + x - prev_u) + coef * du;
          X_grad_ptr[j * H + k] = mask ? T(0) : static_cast<T>(dx);
          u_grad_ptr[k] =
              mask ? u_grad_ptr[k] : (T_ACC(1) - coef) * du - coef * delta * dv;
          v_grad_ptr[k] = mask ? v_grad_ptr[k] : (T_ACC(1) - coef) * dv;
          w_grad_ptr[k] += mask ? T_ACC(0) : dy_rstd * delta;
          b_grad_ptr[k] += mask ? T_ACC(0) : dy;
        }
      }
      for (int64_t j = 0; j < H; ++j) {
        m1_grad_ptr[j] = static_cast<T>(u_grad_ptr[j]);
        m2_grad_ptr[j] = static_cast<T>(v_grad_ptr[j]);
      }
    }
  });

  for (int64_t i = 1; i < B; ++i) {
    for (int64_t j = 0; j < H; ++j) {
      w_grad[j] += w_grad[i * H + j];
      b_grad[j] += b_grad[i * H + j];
    }
  }
  for (int64_t i = 0; i < H; ++i) {
    gamma_grad_data[i] = static_cast<T>(w_grad[i]);
    beta_grad_data[i] = static_cast<T>(b_grad[i]);
  }
}

template <typename T>
void GroupTimestepNormCPUFwdImpl(
    const torch::Tensor& X, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, int64_t num_groups, double eps,
    torch::Tensor& Y, torch::Tensor& count, torch::Tensor& mean,
    torch::Tensor& var, torch::Tensor& group_mean, torch::Tensor& group_var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t D = H / num_groups;

  const T* X_data = X.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();
  T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();

  at::parallel_for(0, B * L * num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const auto [m0, m1, m2] = utils::RowwiseMoments<T, T_ACC>(
          D, X_data + i * D, /*padding_mask=*/nullptr);
      group_mean_data[i] = m1;
      group_var_data[i] = m2;
    }
  });

  std::vector<T_ACC> u(B * num_groups, T_ACC(0));
  std::vector<T_ACC> v(B * num_groups, T_ACC(0));

  at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* X_ptr = X_data + i * L * H;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + i * L;
      const T_ACC* gu_ptr = group_mean_data + i * L * num_groups;
      const T_ACC* gv_ptr = group_var_data + i * L * num_groups;

      T* Y_ptr = Y_data + i * L * H;
      T* m1_ptr = mean_data + i * num_groups;
      T* m2_ptr = var_data + i * num_groups;
      T_ACC* cu_ptr = cummean_data + i * L * num_groups;
      T_ACC* cr_ptr = cumrstd_data + i * L * num_groups;
      T_ACC* u_ptr = u.data() + i * num_groups;
      T_ACC* v_ptr = v.data() + i * num_groups;

      count_data[i] = prev_count_data[i];
      int64_t m0 = prev_count_data[i];
      for (int64_t g = 0; g < num_groups; ++g) {
        u_ptr[g] = static_cast<T_ACC>(prev_mean_data[i * num_groups + g]);
        v_ptr[g] = static_cast<T_ACC>(prev_var_data[i * num_groups + g]);
      }

      for (int64_t j = 0; j < L; ++j) {
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        for (int64_t g = 0; g < num_groups; ++g) {
          const auto [_, m1, m2] = utils::WelfordCombine(
              m0, u_ptr[g], v_ptr[g], 1, gu_ptr[j * num_groups + g],
              gv_ptr[j * num_groups + g]);
          const T_ACC rstd = T(1) / std::sqrt(m2 + static_cast<T_ACC>(eps));
          u_ptr[g] = mask ? u_ptr[g] : m1;
          v_ptr[g] = mask ? v_ptr[g] : m2;
          cu_ptr[j * num_groups + g] = m1;
          cr_ptr[j * num_groups + g] = rstd;
          for (int64_t k = 0; k < D; ++k) {
            const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + g * D + k]);
            const T_ACC w = static_cast<T_ACC>(gamma_data[g * D + k]);
            const T_ACC b = static_cast<T_ACC>(beta_data[g * D + k]);
            Y_ptr[j * H + g * D + k] =
                mask ? T(0) : static_cast<T>((x - m1) * rstd * w + b);
          }
        }
        m0 += mask ? 0 : 1;
        count_data[i] = m0;
      }

      for (int64_t g = 0; g < num_groups; ++g) {
        m1_ptr[g] = static_cast<T>(u_ptr[g]);
        m2_ptr[g] = static_cast<T>(v_ptr[g]);
      }
    }
  });
}

template <typename T>
void GroupTimestepNormCPUBwdImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& padding_mask,
    int64_t num_groups, torch::Tensor& X_grad, torch::Tensor& prev_mean_grad,
    torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t D = H / num_groups;
  const T_ACC cg = T_ACC(1) / T_ACC(D);

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  const T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* m1_grad_data = prev_mean_grad.data_ptr<T>();
  T* m2_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();

  std::vector<T_ACC> u_grad(B * num_groups, T_ACC(0));
  std::vector<T_ACC> v_grad(B * num_groups, T_ACC(0));
  std::vector<T_ACC> w_grad(B * H, T_ACC(0));
  std::vector<T_ACC> b_grad(B * H, T_ACC(0));

  at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* Y_grad_ptr = Y_grad_data + i * L * H;
      const T* mean_grad_ptr = mean_grad_data + i * num_groups;
      const T* var_grad_ptr = var_grad_data + i * num_groups;
      const T* X_ptr = X_data + i * L * H;
      const T_ACC* mean_ptr = cummean_data + i * L * num_groups;
      const T_ACC* rstd_ptr = cumrstd_data + i * L * num_groups;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + i * L;

      T* X_grad_ptr = X_grad_data + i * L * H;
      T* m1_grad_ptr = m1_grad_data + i * num_groups;
      T* m2_grad_ptr = m2_grad_data + i * num_groups;

      T_ACC* u_grad_ptr = u_grad.data() + i * num_groups;
      T_ACC* v_grad_ptr = v_grad.data() + i * num_groups;
      T_ACC* w_grad_ptr = w_grad.data() + i * H;
      T_ACC* b_grad_ptr = b_grad.data() + i * H;

      int64_t m0 = count_data[i];
      for (int64_t g = 0; g < num_groups; ++g) {
        u_grad_ptr[g] = static_cast<T_ACC>(mean_grad_ptr[g]);
        v_grad_ptr[g] = static_cast<T_ACC>(var_grad_ptr[g]);
      }

      for (int64_t j = L - 1; j >= 0; --j) {
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        const T_ACC c1 = static_cast<T_ACC>(m0 - 1) / static_cast<T_ACC>(m0);
        const T_ACC c2 = T_ACC(1) / static_cast<T_ACC>(m0);
        m0 -= mask ? 0 : 1;

        for (int64_t g = 0; g < num_groups; ++g) {
          const T_ACC prev_u =
              j == 0 ? static_cast<T_ACC>(prev_mean_data[i * num_groups + g])
                     : mean_ptr[(j - 1) * num_groups + g];
          // const T_ACC ux = ux_ptr[j * num_groups + g];
          const T_ACC u = mean_ptr[j * num_groups + g];
          const T_ACC r = rstd_ptr[j * num_groups + g];

          T_ACC ux = T_ACC(0);
          T_ACC ds = T_ACC(0);
          T_ACC db = T_ACC(0);
          for (int64_t k = 0; k < D; ++k) {
            const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + g * D + k]);
            const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + g * D + k]);
            const T_ACC w = static_cast<T_ACC>(gamma_data[g * D + k]);
            ux += x;
            ds += dy * (x - u) * w;
            db += dy * w;
          }
          ux /= static_cast<T_ACC>(D);

          const T_ACC du = u_grad_ptr[g] - r * db;
          const T_ACC dv = v_grad_ptr[g] - T_ACC(0.5) * utils::Cube(r) * ds;
          const T_ACC dux = c2 * du + T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
          const T_ACC dvx = c2 * dv;
          for (int64_t k = 0; k < D; ++k) {
            const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + g * D + k]);
            const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + g * D + k]);
            const T_ACC w = static_cast<T_ACC>(gamma_data[g * D + k]);
            const T_ACC dx =
                dy * r * w + dux * cg + T_ACC(2) * dvx * cg * (x - ux);
            X_grad_ptr[j * H + g * D + k] = mask ? T(0) : static_cast<T>(dx);
            w_grad_ptr[g * D + k] += mask ? T_ACC(0) : dy * (x - u) * r;
            b_grad_ptr[g * D + k] += mask ? T_ACC(0) : dy;
          }
          u_grad_ptr[g] =
              mask ? u_grad_ptr[g]
                   : c1 * du - T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
          v_grad_ptr[g] = mask ? v_grad_ptr[g] : c1 * dv;
        }
      }

      for (int64_t g = 0; g < num_groups; ++g) {
        m1_grad_ptr[g] = static_cast<T>(u_grad_ptr[g]);
        m2_grad_ptr[g] = static_cast<T>(v_grad_ptr[g]);
      }
    }
  });

  for (int64_t i = 1; i < B; ++i) {
    for (int64_t j = 0; j < H; ++j) {
      w_grad[j] += w_grad[i * H + j];
      b_grad[j] += b_grad[i * H + j];
    }
  }
  for (int64_t i = 0; i < H; ++i) {
    gamma_grad_data[i] = static_cast<T>(w_grad[i]);
    beta_grad_data[i] = static_cast<T>(b_grad[i]);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCPUFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                   const torch::Tensor& prev_mean,
                   const torch::Tensor& prev_var, const torch::Tensor& gamma,
                   const torch::Tensor& beta,
                   const c10::optional<torch::Tensor>& padding_mask,
                   double eps) {
  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count,
      prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(
      prev_mean,
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty(
      {B, L, H},
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty(
      {B, L, H},
      prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "TimestepNormCPUFwd", [&]() {
        TimestepNormCPUFwdImpl<scalar_t>(
            *(X.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
            mean, var, cummean, cumrstd);
      });
  return std::make_tuple(Y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
                const torch::Tensor& gamma, const torch::Tensor& beta,
                const c10::optional<torch::Tensor>& padding_mask, double eps) {
  if (X.device().type() == torch::kCUDA) {
    return TimestepNormCUDAFwd(X, prev_count, prev_mean, prev_var, gamma, beta,
                               padding_mask, eps);
  } else {
    return TimestepNormCPUFwd(X, prev_count, prev_mean, prev_var, gamma, beta,
                              padding_mask, eps);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCPUBwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                   const torch::Tensor& var_grad, const torch::Tensor& X,
                   const torch::Tensor& prev_mean, const torch::Tensor& count,
                   const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                   const torch::Tensor& gamma,
                   const c10::optional<torch::Tensor>& padding_mask) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_mean_grad = torch::empty_like(
      mean_grad,
      mean_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(
      var_grad, var_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "TimestepNormCPUBwd", [&]() {
        TimestepNormCPUBwdImpl<scalar_t>(
            *(Y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(X.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(count.expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
            prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(X_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormBwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                const torch::Tensor& var_grad, const torch::Tensor& X,
                const torch::Tensor& prev_mean, const torch::Tensor& count,
                const torch::Tensor& mean, const torch::Tensor& var,
                const torch::Tensor& gamma,
                const c10::optional<torch::Tensor>& padding_mask) {
  if (X.device().type() == torch::kCUDA) {
    return TimestepNormCUDABwd(Y_grad, mean_grad, var_grad, X, prev_mean, count,
                               mean, var, gamma, padding_mask);
  } else {
    return TimestepNormCPUBwd(Y_grad, mean_grad, var_grad, X, prev_mean, count,
                              mean, var, gamma, padding_mask);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormCPUFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                        const torch::Tensor& prev_mean,
                        const torch::Tensor& prev_var,
                        const torch::Tensor& gamma, const torch::Tensor& beta,
                        const c10::optional<torch::Tensor>& padding_mask,
                        int64_t num_groups, double eps) {
  const int64_t B = X.size(0);
  const int64_t L = X.size(1);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count,
      prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(
      prev_mean,
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toOpMathType(X.scalar_type());
  torch::Tensor group_mean = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor group_var = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "GroupTimestepNormCPUFwd",
      [&]() {
        GroupTimestepNormCPUFwdImpl<scalar_t>(
            *(X.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups, eps,
            Y, count, mean, var, group_mean, group_var, cummean, cumrstd);
      });

  return std::make_tuple(Y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                     const torch::Tensor& prev_mean,
                     const torch::Tensor& prev_var, const torch::Tensor& gamma,
                     const torch::Tensor& beta,
                     const c10::optional<torch::Tensor>& padding_mask,
                     int64_t num_groups, double eps) {
  if (X.device().type() == torch::kCUDA) {
    return GroupTimestepNormCUDAFwd(X, prev_count, prev_mean, prev_var, gamma,
                                    beta, padding_mask, num_groups, eps);
  } else {
    return GroupTimestepNormCPUFwd(X, prev_count, prev_mean, prev_var, gamma,
                                   beta, padding_mask, num_groups, eps);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormCPUBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_mean_grad = torch::empty_like(
      mean_grad,
      mean_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(
      var_grad, var_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "GroupTimestepNormCPUBwd",
      [&]() {
        GroupTimestepNormCPUBwdImpl<scalar_t>(
            *(Y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(X.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(count.expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups,
            X_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(X_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormBwd(const torch::Tensor& Y_grad,
                     const torch::Tensor& mean_grad,
                     const torch::Tensor& var_grad, const torch::Tensor& X,
                     const torch::Tensor& prev_mean, const torch::Tensor& count,
                     const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& padding_mask,
                     int64_t num_groups) {
  if (X.device().type() == torch::kCUDA) {
    return GroupTimestepNormCUDABwd(Y_grad, mean_grad, var_grad, X, prev_mean,
                                    count, cummean, cumrstd, gamma,
                                    padding_mask, num_groups);
  } else {
    return GroupTimestepNormCPUBwd(Y_grad, mean_grad, var_grad, X, prev_mean,
                                   count, cummean, cumrstd, gamma, padding_mask,
                                   num_groups);
  }
}

void DefineTimestepNormOp(py::module& m) {
  m.def("timestep_norm_fwd", &TimestepNormFwd, "TimestepNorm forward")
      .def("timestep_norm_bwd", &TimestepNormBwd, "TimestepNorm backward")
      .def("group_timestep_norm_fwd", &GroupTimestepNormFwd,
           "GroupTimestepNorm forward")
      .def("group_timestep_norm_bwd", &GroupTimestepNormBwd,
           "GroupTimestepNorm backward");
}

}  // namespace ops
}  // namespace megalodon
