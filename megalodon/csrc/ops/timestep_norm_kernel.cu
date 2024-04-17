#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <thrust/tuple.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <cstring>
#include <tuple>
#include <vector>

#include "cuda_utils.cuh"
#include "kahan.h"
#include "ops/timestep_norm.h"
#include "reduce.cuh"
#include "register_utils.cuh"
#include "welford.h"

namespace megalodon {
namespace ops {

namespace {

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDAFwdSmallKernel(
    int64_t L, int64_t H, const T* __restrict__ x,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean,
    const T* __restrict__ prev_var, const T* __restrict__ gamma,
    const T* __restrict__ beta, const bool* __restrict__ padding_mask,
    T_ACC eps, T* __restrict__ y, int64_t* __restrict__ count,
    T* __restrict__ mean, T* __restrict__ var, T* __restrict__ cummean,
    T* __restrict__ cumrstd) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* x_ptr = x + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T* y_ptr = y + i * L * H;
  T* m1_ptr = mean + i * H;
  T* m2_ptr = var + i * H;
  T* cu_ptr = cummean + i * L * H;
  T* cr_ptr = cumrstd + i * L * H;

  const T_ACC w_acc = static_cast<T_ACC>(gamma[k]);
  const T_ACC b_acc = static_cast<T_ACC>(beta[k]);

  const int64_t m0 = prev_count[i];
  const T_ACC m1 = static_cast<T_ACC>(prev_mean[i * H + k]);
  const T_ACC m2 = static_cast<T_ACC>(prev_var[i * H + k]);
  utils::KahanWrapper<utils::WelfordData<T_ACC>> m(
      utils::WelfordData<T_ACC>{m0, m1, m2});

  // TODO: Improve this.
  for (int64_t j = 0; j < L; ++j) {
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];

    const utils::KahanWrapper<utils::WelfordData<T_ACC>> nxt = m + x_acc;
    m = mask ? m : nxt;
    // m = mask ? m : m + x_acc;
    const T_ACC rstd = c10::cuda::compat::rsqrt(m->m2 + eps);
    y_ptr[j * H + k] =
        mask ? T(0) : static_cast<T>((x_acc - m->m1) * rstd * w_acc + b_acc);
    cu_ptr[j * H + k] = static_cast<T>(m->m1);
    cr_ptr[j * H + k] = static_cast<T>(rstd);
  }
  if (k == 0) {
    count[i] = m->m0;
  }
  m1_ptr[k] = static_cast<T>(m->m1);
  m2_ptr[k] = static_cast<T>(m->m2);
}

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDAFwdLargeKernel(
    int64_t L, int64_t H, const int64_t chunk_size, const T* __restrict__ x,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean,
    const T* __restrict__ prev_var, const T* __restrict__ gamma,
    const T* __restrict__ beta, const bool* __restrict__ padding_mask,
    T_ACC eps, T* __restrict__ y, int64_t* __restrict__ count,
    T* __restrict__ mean, T* __restrict__ var, T* __restrict__ cummean,
    T* __restrict__ cumrstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData
      shm[cuda_utils::kWarpSize * cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (k >= H) {
    return;
  }

  const T* x_ptr = x + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T* y_ptr = y + i * L * H;
  T* m1_ptr = mean + i * H;
  T* m2_ptr = var + i * H;
  T* cu_ptr = cummean + i * L * H;
  T* cr_ptr = cumrstd + i * L * H;

  const T_ACC w_acc = static_cast<T_ACC>(gamma[k]);
  const T_ACC b_acc = static_cast<T_ACC>(beta[k]);

  utils::KahanWrapper<utils::WelfordData<T_ACC>> m(utils::WelfordData<T_ACC>{});
  for (int64_t j = l; j < r; ++j) {
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const utils::KahanWrapper<utils::WelfordData<T_ACC>> nxt = m + x_acc;
    m = mask ? m : nxt;
    // m = mask ? m : m + x_acc;
  }
  shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x] = *m;
  __syncthreads();

  int64_t offset = 1;
#pragma unroll
  for (int64_t d = cuda_utils::kWarpSize >> 1; d > 0; d >>= 1) {
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x] +=
          shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    shm_ptr[(cuda_utils::kWarpSize - 1) * cuda_utils::kWarpSize + threadIdx.x] =
        utils::WelfordData<T_ACC>{prev_count[i],
                                  static_cast<T_ACC>(prev_mean[i * H + k]),
                                  static_cast<T_ACC>(prev_var[i * H + k])};
  }
  __syncthreads();
#pragma unroll
  for (int64_t d = 1; d < cuda_utils::kWarpSize; d <<= 1) {
    offset >>= 1;
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      const utils::WelfordData<T_ACC> c =
          shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x];
      shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x] =
          shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x];
      shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x] += c;
    }
    __syncthreads();
  }

  m = utils::KahanWrapper<utils::WelfordData<T_ACC>>(
      shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x]);
  for (int64_t j = l; j < r; ++j) {
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const utils::KahanWrapper<utils::WelfordData<T_ACC>> nxt = m + x_acc;
    m = mask ? m : nxt;
    // m = mask ? m : m + x_acc;
    const T_ACC rstd = c10::cuda::compat::rsqrt(m->m2 + eps);
    y_ptr[j * H + k] =
        mask ? T(0) : static_cast<T>((x_acc - m->m1) * rstd * w_acc + b_acc);
    cu_ptr[j * H + k] = static_cast<T>(m->m1);
    cr_ptr[j * H + k] = static_cast<T>(rstd);
  }
  if (threadIdx.y == cuda_utils::kWarpSize - 1) {
    if (k == 0) {
      count[i] = m->m0;
    }
    m1_ptr[k] = static_cast<T>(m->m1);
    m2_ptr[k] = static_cast<T>(m->m2);
  }
}

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDABwdSmallKernel(
    int64_t L, int64_t H, const T* __restrict__ y_grad,
    const T* __restrict__ mean_grad, const T* __restrict__ var_grad,
    const T* __restrict__ x, const T* __restrict__ prev_mean,
    const int64_t* __restrict__ count, const T* __restrict__ cummean,
    const T* __restrict__ cumrstd, const T* __restrict__ gamma,
    const bool* __restrict__ padding_mask, T* __restrict__ x_grad,
    T* __restrict__ prev_mean_grad, T* __restrict__ prev_var_grad,
    T_ACC* __restrict__ gamma_grad, T_ACC* __restrict__ beta_grad) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* y_grad_ptr = y_grad + i * L * H;
  const T* mean_grad_ptr = mean_grad + i * H;
  const T* var_grad_ptr = var_grad + i * H;
  const T* x_ptr = x + i * L * H;
  const T* mean_ptr = cummean + i * L * H;
  const T* rstd_ptr = cumrstd + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;

  T* x_grad_ptr = x_grad + i * L * H;
  T* m1_grad_ptr = prev_mean_grad + i * H;
  T* m2_grad_ptr = prev_var_grad + i * H;
  T_ACC* w_grad_ptr = gamma_grad + i * H;
  T_ACC* b_grad_ptr = beta_grad + i * H;

  const T_ACC w_acc = static_cast<T_ACC>(gamma[k]);

  int64_t m0 = count[i];
  T_ACC u_grad = static_cast<T_ACC>(mean_grad_ptr[k]);
  T_ACC v_grad = static_cast<T_ACC>(var_grad_ptr[k]);
  utils::KahanWrapper<T_ACC> sum1(T_ACC(0));
  utils::KahanWrapper<T_ACC> sum2(T_ACC(0));
  // TODO: Improve this.
  for (int64_t j = L - 1; j >= 0; --j) {
    const T_ACC dy_acc = static_cast<T_ACC>(y_grad_ptr[j * H + k]);
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const T_ACC prev_u = static_cast<T_ACC>(j == 0 ? prev_mean[i * H + k]
                                                   : mean_ptr[(j - 1) * H + k]);
    const T_ACC u = static_cast<T_ACC>(mean_ptr[j * H + k]);
    const T_ACC r = static_cast<T_ACC>(rstd_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(m0);
    const T_ACC dy_rstd = dy_acc * r;
    const T_ACC delta1 = x_acc - prev_u;
    const T_ACC delta2 = x_acc - u;
    const T_ACC dv =
        v_grad - (T_ACC(0.5) * dy_acc * w_acc * delta2 * cuda_utils::Cube(r));
    const T_ACC du = u_grad - (w_acc * dy_rstd + coef * delta1 * dv);
    const T_ACC dx =
        w_acc * dy_rstd + dv * coef * (delta1 + delta2) + coef * du;
    x_grad_ptr[j * H + k] = mask ? T(0) : static_cast<T>(dx);
    u_grad = mask ? u_grad : (T_ACC(1) - coef) * du - coef * delta2 * dv;
    v_grad = mask ? v_grad : (T_ACC(1) - coef) * dv;

    // sum1 += mask ? T_ACC(0) : dy_rstd * delta2;
    // sum2 += mask ? T_ACC(0) : dy_acc;
    const utils::KahanWrapper<T_ACC> t1 = sum1 + dy_rstd * delta2;
    const utils::KahanWrapper<T_ACC> t2 = sum2 + dy_acc;
    sum1 = mask ? sum1 : t1;
    sum2 = mask ? sum2 : t2;
    m0 -= mask ? 0 : 1;
  }

  w_grad_ptr[k] = *sum1;
  b_grad_ptr[k] = *sum2;
  m1_grad_ptr[k] = static_cast<T>(u_grad);
  m2_grad_ptr[k] = static_cast<T>(v_grad);
}

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDABwdLargeKernel(
    int64_t L, int64_t H, int64_t chunk_size, const T* __restrict__ y_grad,
    const T* __restrict__ mean_grad, const T* __restrict__ var_grad,
    const T* __restrict__ x, const T* __restrict__ prev_mean,
    const int64_t* __restrict__ count, const T* __restrict__ cummean,
    const T* __restrict__ cumrstd, const T* __restrict__ gamma,
    const bool* __restrict__ padding_mask, T* __restrict__ x_grad,
    T* __restrict__ prev_mean_grad, T* __restrict__ prev_var_grad,
    T_ACC* __restrict__ gamma_grad, T_ACC* __restrict__ beta_grad) {
  __shared__ int64_t
      m0_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m1_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC du_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC dv_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (k >= H) {
    return;
  }

  const T* y_grad_ptr = y_grad + i * L * H;
  const T* mean_grad_ptr = mean_grad + i * H;
  const T* var_grad_ptr = var_grad + i * H;
  const T* x_ptr = x + i * L * H;
  const T* mean_ptr = cummean + i * L * H;
  const T* rstd_ptr = cumrstd + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;

  T* x_grad_ptr = x_grad + i * L * H;
  T* m1_grad_ptr = prev_mean_grad + i * H;
  T* m2_grad_ptr = prev_var_grad + i * H;
  T_ACC* w_grad_ptr = gamma_grad + i * H;
  T_ACC* b_grad_ptr = beta_grad + i * H;

  const T_ACC w_acc = static_cast<T_ACC>(gamma[k]);

  int64_t cnt = 0;
  utils::KahanWrapper<T_ACC> sum(T_ACC(0));
  for (int64_t j = l; j < r; ++j) {
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    cnt += mask ? 0 : 1;
    // sum += mask ? T_ACC(0) : x_acc;
    const utils::KahanWrapper<T_ACC> nxt = sum + x_acc;
    sum = mask ? sum : nxt;
  }
  m0_shared[threadIdx.y][threadIdx.x] = cnt;
  m1_shared[threadIdx.y][threadIdx.x] =
      cnt == 0 ? T_ACC(0) : *sum / static_cast<T_ACC>(cnt);
  __syncthreads();

  if (threadIdx.y == 0) {
    int64_t m0 = count[i];
#pragma unroll
    for (int64_t j = cuda_utils::kWarpSize / 2 - 1; j >= 0; --j) {
      const int64_t cur = m0_shared[j][threadIdx.x];
      m0_shared[j][threadIdx.x] = m0;
      m0 -= cur;
    }
  }
  __syncthreads();

  int64_t m0 = m0_shared[threadIdx.y][threadIdx.x];
  T_ACC u_grad = T_ACC(0);
  T_ACC v_grad = T_ACC(0);
  for (int64_t j = r - 1; j >= l; --j) {
    const T_ACC dy_acc = static_cast<T_ACC>(y_grad_ptr[j * H + k]);
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const T_ACC prev_u = static_cast<T_ACC>(j == 0 ? prev_mean[i * H + k]
                                                   : mean_ptr[(j - 1) * H + k]);
    const T_ACC u = static_cast<T_ACC>(mean_ptr[j * H + k]);
    const T_ACC r = static_cast<T_ACC>(rstd_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(m0);
    const T_ACC dy_rstd = dy_acc * r;
    const T_ACC delta1 = x_acc - prev_u;
    const T_ACC delta2 = x_acc - u;
    const T_ACC dv =
        v_grad - (T_ACC(0.5) * dy_acc * w_acc * delta2 * cuda_utils::Cube(r));
    const T_ACC du = u_grad - (w_acc * dy_rstd + coef * delta1 * dv);
    u_grad = mask ? u_grad : (T_ACC(1) - coef) * du - coef * delta2 * dv;
    v_grad = mask ? v_grad : (T_ACC(1) - coef) * dv;
    m0 -= mask ? 0 : 1;
  }
  du_shared[threadIdx.y][threadIdx.x] = u_grad;
  dv_shared[threadIdx.y][threadIdx.x] = v_grad;
  __syncthreads();

  if (threadIdx.y == 0) {
    T_ACC du = static_cast<T_ACC>(mean_grad_ptr[k]);
    T_ACC dv = static_cast<T_ACC>(var_grad_ptr[k]);
#pragma unroll
    for (int64_t j = cuda_utils::kWarpSize / 2 - 1; j >= 0; --j) {
      const T_ACC prev_u =
          static_cast<T_ACC>(j == 0 ? prev_mean[i * H + k]
                                    : mean_ptr[(j * chunk_size - 1) * H + k]);
      const int64_t n = m0_shared[j][threadIdx.x];
      const int64_t m = j == 0 ? n - cnt : m0_shared[j - 1][threadIdx.x];
      const T_ACC c1 = static_cast<T_ACC>(m) / static_cast<T_ACC>(n);
      const T_ACC c2 = T_ACC(1) - c1;
      const T_ACC m1x = m1_shared[j][threadIdx.x];
      const T_ACC dux = du_shared[j][threadIdx.x];
      const T_ACC dvx = dv_shared[j][threadIdx.x];
      du_shared[j][threadIdx.x] = du;
      dv_shared[j][threadIdx.x] = dv;
      du = dux + c1 * du - T_ACC(2) * c1 * c2 * dv * (m1x - prev_u);
      dv = dvx + c1 * dv;
    }
  }
  __syncthreads();

  m0 = m0_shared[threadIdx.y][threadIdx.x];
  u_grad = du_shared[threadIdx.y][threadIdx.x];
  v_grad = dv_shared[threadIdx.y][threadIdx.x];
  utils::KahanWrapper<T_ACC> sum1(T_ACC(0));
  utils::KahanWrapper<T_ACC> sum2(T_ACC(0));
  for (int64_t j = r - 1; j >= l; --j) {
    const T_ACC dy_acc = static_cast<T_ACC>(y_grad_ptr[j * H + k]);
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[j * H + k]);
    const T_ACC prev_u = static_cast<T_ACC>(j == 0 ? prev_mean[i * H + k]
                                                   : mean_ptr[(j - 1) * H + k]);
    const T_ACC u = static_cast<T_ACC>(mean_ptr[j * H + k]);
    const T_ACC r = static_cast<T_ACC>(rstd_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(m0);
    const T_ACC dy_rstd = dy_acc * r;
    const T_ACC delta1 = x_acc - prev_u;
    const T_ACC delta2 = x_acc - u;
    const T_ACC dv =
        v_grad - (T_ACC(0.5) * dy_acc * w_acc * delta2 * cuda_utils::Cube(r));
    const T_ACC du = u_grad - (w_acc * dy_rstd + coef * delta1 * dv);
    const T_ACC dx =
        w_acc * dy_rstd + dv * coef * (delta1 + delta2) + coef * du;
    x_grad_ptr[j * H + k] = mask ? T(0) : static_cast<T>(dx);
    // sum1 += mask ? T_ACC(0) : dy_rstd * delta2;
    // sum2 += mask ? T_ACC(0) : dy_acc;
    const utils::KahanWrapper<T_ACC> t1 = sum1 + dy_rstd * delta2;
    const utils::KahanWrapper<T_ACC> t2 = sum2 + dy_acc;
    sum1 = mask ? sum1 : t1;
    sum2 = mask ? sum2 : t2;
    u_grad = mask ? u_grad : (T_ACC(1) - coef) * du - coef * delta2 * dv;
    v_grad = mask ? v_grad : (T_ACC(1) - coef) * dv;
    m0 -= mask ? 0 : 1;
  }
  du_shared[threadIdx.y][threadIdx.x] = *sum1;
  dv_shared[threadIdx.y][threadIdx.x] = *sum2;
  __syncthreads();

  if (threadIdx.y == 0) {
    m1_grad_ptr[k] = static_cast<T>(u_grad);
    m2_grad_ptr[k] = static_cast<T>(v_grad);

    T_ACC s1 = T_ACC(0);
    T_ACC s2 = T_ACC(0);
#pragma unroll
    for (int64_t j = 0; j < cuda_utils::kWarpSize / 2; ++j) {
      s1 += du_shared[j][threadIdx.x];
      s2 += dv_shared[j][threadIdx.x];
    }
    w_grad_ptr[k] = s1;
    b_grad_ptr[k] = s2;
  }
}

template <typename T, typename T_ACC>
__global__ void GammaBetaCUDABwdKernel(int64_t B, int64_t H,
                                       const T_ACC* __restrict__ dw_internal,
                                       const T_ACC* __restrict__ db_internal,
                                       T* __restrict__ dw, T* __restrict__ db) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) {
    return;
  }
  T_ACC w_grad = T_ACC(0);
  T_ACC b_grad = T_ACC(0);
  for (int64_t i = 0; i < B; ++i) {
    w_grad += dw_internal[i * H + j];
    b_grad += db_internal[i * H + j];
  }
  dw[j] = static_cast<T>(w_grad);
  db[j] = static_cast<T>(b_grad);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void RowwiseMomentsKernel(int64_t size, const T* __restrict__ x,
                                     T_ACC* __restrict__ mean,
                                     T_ACC* __restrict__ var) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  const int64_t r = blockIdx.x;
  __shared__ T_ACC shm[cuda_utils::kWarpSize];

  T_ACC x_acc[kElementsPerThread];
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + r * size, size, size, std::numeric_limits<T_ACC>::infinity(), x_acc);

  const T_ACC coef = T(1) / static_cast<T_ACC>(size);
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    m1 += isinf(x_acc[i]) ? T_ACC(0) : x_acc[i];
  }
  m1 = reduce::BlockAllReduce(m1, shm);
  m1 *= coef;

#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    m2 += isinf(x_acc[i]) ? T_ACC(0) : cuda_utils::Square<T_ACC>(x_acc[i] - m1);
  }
  m2 = reduce::BlockReduce(m2, shm);

  if (threadIdx.x == 0) {
    mean[r] = m1;
    var[r] = m2 * coef;
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseCumMomentsSmallKernel(
    int64_t L, int64_t num_groups, const int64_t* __restrict__ prev_count,
    const T* __restrict__ prev_mean, const T* __restrict__ prev_var,
    const T_ACC* __restrict__ group_mean, const T_ACC* __restrict__ group_var,
    const bool* __restrict__ padding_mask, T_ACC eps,
    int64_t* __restrict__ count, T* __restrict__ mean, T* __restrict__ var,
    T_ACC* __restrict__ cummean, T_ACC* __restrict__ cumrstd) {
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_groups) {
    return;
  }

  const T_ACC* gu_ptr = group_mean + b * L * num_groups;
  const T_ACC* gv_ptr = group_var + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T_ACC* cummean_ptr = cummean + b * L * num_groups;
  T_ACC* cumrstd_ptr = cumrstd + b * L * num_groups;

  const int64_t m0 = prev_count[b];
  const T_ACC m1 = static_cast<T_ACC>(prev_mean[b * num_groups + g]);
  const T_ACC m2 = static_cast<T_ACC>(prev_var[b * num_groups + g]);
  utils::KahanWrapper<utils::WelfordData<T_ACC>> m(
      utils::WelfordData<T_ACC>{m0, m1, m2});

  for (int64_t i = 0; i < L; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordData<T_ACC> cur = {1, gu, gv};
    const utils::KahanWrapper<utils::WelfordData<T_ACC>> nxt = m + cur;
    m = mask ? m : nxt;
    // m = mask ? m : m + cur;
    const T_ACC rstd = c10::cuda::compat::rsqrt(m->m2 + eps);
    cummean_ptr[i * num_groups + g] = m->m1;
    cumrstd_ptr[i * num_groups + g] = rstd;
  }

  if (g == 0) {
    count[b] = m->m0;
  }
  mean[b * num_groups + g] = static_cast<T>(m->m1);
  var[b * num_groups + g] = static_cast<T>(m->m2);
}

template <typename T, typename T_ACC>
__global__ void ColwiseCumMomentsLargeKernel(
    int64_t L, int64_t num_groups, int64_t chunk_size,
    const int64_t* __restrict__ prev_count, const T* __restrict__ prev_mean,
    const T* __restrict__ prev_var, const T_ACC* __restrict__ group_mean,
    const T_ACC* __restrict__ group_var, const bool* __restrict__ padding_mask,
    T_ACC eps, int64_t* __restrict__ count, T* __restrict__ mean,
    T* __restrict__ var, T_ACC* __restrict__ cummean,
    T_ACC* __restrict__ cumrstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData
      shm[cuda_utils::kWarpSize * cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (g >= num_groups) {
    return;
  }

  const T_ACC* gu_ptr = group_mean + b * L * num_groups;
  const T_ACC* gv_ptr = group_var + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T_ACC* cummean_ptr = cummean + b * L * num_groups;
  T_ACC* cumrstd_ptr = cumrstd + b * L * num_groups;

  utils::KahanWrapper<utils::WelfordData<T_ACC>> m(utils::WelfordData<T_ACC>{});
  for (int64_t i = l; i < r; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordData<T_ACC> cur = {1, gu, gv};
    const utils::KahanWrapper<utils::WelfordData<T_ACC>> nxt = m + cur;
    m = mask ? m : nxt;
    // m = mask ? m : m + cur;
  }
  shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x] = *m;
  __syncthreads();

  int64_t offset = 1;
#pragma unroll
  for (int64_t d = cuda_utils::kWarpSize >> 1; d > 0; d >>= 1) {
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x] +=
          shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    shm_ptr[(cuda_utils::kWarpSize - 1) * cuda_utils::kWarpSize + threadIdx.x] =
        utils::WelfordData<T_ACC>{
            prev_count[b], static_cast<T_ACC>(prev_mean[b * num_groups + g]),
            static_cast<T_ACC>(prev_var[b * num_groups + g])};
  }
  __syncthreads();
#pragma unroll
  for (int64_t d = 1; d < cuda_utils::kWarpSize; d <<= 1) {
    offset >>= 1;
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      const utils::WelfordData<T_ACC> c =
          shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x];
      shm_ptr[ai * cuda_utils::kWarpSize + threadIdx.x] =
          shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x];
      shm_ptr[bi * cuda_utils::kWarpSize + threadIdx.x] += c;
    }
    __syncthreads();
  }

  m = utils::KahanWrapper<utils::WelfordData<T_ACC>>(
      shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x]);
  for (int64_t i = l; i < r; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordData<T_ACC> cur = {1, gu, gv};
    const utils::KahanWrapper<utils::WelfordData<T_ACC>> nxt = m + cur;
    m = mask ? m : nxt;
    // m = mask ? m : m + cur;
    const T_ACC rstd = c10::cuda::compat::rsqrt(m->m2 + eps);
    cummean_ptr[i * num_groups + g] = m->m1;
    cumrstd_ptr[i * num_groups + g] = rstd;
  }

  if (threadIdx.y == cuda_utils::kWarpSize - 1) {
    if (g == 0) {
      count[b] = m->m0;
    }
    mean[b * num_groups + g] = static_cast<T>(m->m1);
    var[b * num_groups + g] = static_cast<T>(m->m2);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupTimestepNormCUDAFwdKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const T* __restrict__ gamma, const T* __restrict__ beta,
    const bool* __restrict__ padding_mask, T* __restrict__ y) {
  extern __shared__ float shm[];

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t l = blockIdx.x;

  T_ACC* u_shared = reinterpret_cast<T_ACC*>(shm);
  T_ACC* r_shared = u_shared + num_groups;

  const T* x_ptr = x + (b * L + l) * H;
  const T_ACC* cummean_ptr = cummean + (b * L + l) * num_groups;
  const T_ACC* cumrstd_ptr = cumrstd + (b * L + l) * num_groups;
  const bool mask = padding_mask != nullptr && padding_mask[b * L + l];
  T* y_ptr = y + (b * L + l) * H;

  if (mask) {
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
      y_ptr[i] = T(0);
    }
    return;
  }

  for (int64_t i = threadIdx.x; i < num_groups; i += blockDim.x) {
    u_shared[i] = cummean_ptr[i];
    r_shared[i] = cumrstd_ptr[i];
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t g = i / D;
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[i]);
    const T_ACC u = u_shared[g];
    const T_ACC r = r_shared[g];
    const T_ACC w_acc = static_cast<T_ACC>(gamma[i]);
    const T_ACC b_acc = static_cast<T_ACC>(beta[i]);
    y_ptr[i] = static_cast<T>((x_acc - u) * r * w_acc + b_acc);
  }
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void RowwiseInternalGradientsKernel(
    int64_t num_groups, int64_t group_size, const T* __restrict__ y_grad,
    const T* __restrict__ x, const T_ACC* __restrict__ mean,
    const T* __restrict__ gamma, T_ACC* __restrict__ group_mean,
    T_ACC* __restrict__ ds, T_ACC* __restrict__ db) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  const int64_t r = blockIdx.x;
  const int64_t g = blockIdx.y;
  extern __shared__ float shm[];

  T_ACC* m1_shared = reinterpret_cast<T_ACC*>(shm);
  T_ACC* ds_shared = m1_shared + cuda_utils::kWarpSize;
  T_ACC* db_shared = ds_shared + cuda_utils::kWarpSize;
  T_ACC* dy_shared = db_shared + cuda_utils::kWarpSize;
  T_ACC* w_shared = dy_shared + kBlockSize;
  T_ACC x_acc[kElementsPerThread];

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + (r * num_groups + g) * group_size, group_size, group_size,
      T_ACC(0), x_acc);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreads + threadIdx.x;
    dy_shared[idx] = x_acc[i];
  }
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      gamma + g * group_size, group_size, group_size, T_ACC(0), x_acc);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreads + threadIdx.x;
    w_shared[idx] = x_acc[i];
  }
  __syncthreads();

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + (r * num_groups + g) * group_size, group_size, group_size, T_ACC(0),
      x_acc);

  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(group_size);
  const T_ACC u = mean[r * num_groups + g];
  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  T_ACC sum3 = T_ACC(0);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * kNumThreads + threadIdx.x;
    const T_ACC dy_acc = dy_shared[idx];
    const T_ACC w_acc = w_shared[idx];
    sum1 += x_acc[i];
    sum2 += dy_acc * (x_acc[i] - u) * w_acc;
    sum3 += dy_acc * w_acc;
  }
  sum1 = reduce::BlockReduce(sum1, m1_shared);
  sum2 = reduce::BlockReduce(sum2, ds_shared);
  sum3 = reduce::BlockReduce(sum3, db_shared);
  if (threadIdx.x == 0) {
    group_mean[r * num_groups + g] = sum1 * coef;
    ds[r * num_groups + g] = sum2;
    db[r * num_groups + g] = sum3;
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsSmallKernel(
    int64_t L, int64_t num_groups, const T* __restrict__ mean_grad,
    const T* __restrict__ var_grad, const T* __restrict__ prev_mean,
    const T_ACC* __restrict__ group_mean, const int64_t* __restrict__ count,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, const T_ACC* ds, const T_ACC* db,
    T* __restrict__ prev_mean_grad, T* __restrict__ prev_var_grad, T_ACC* du,
    T_ACC* dv) {
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_groups) {
    return;
  }

  const T* mean_grad_ptr = mean_grad + b * num_groups;
  const T* var_grad_ptr = var_grad + b * num_groups;
  const T_ACC* group_mean_ptr = group_mean + b * L * num_groups;
  const T_ACC* mean_ptr = cummean + b * L * num_groups;
  const T_ACC* rstd_ptr = cumrstd + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC* ds_ptr = ds + b * L * num_groups;
  const T_ACC* db_ptr = db + b * L * num_groups;

  T* m1_grad_ptr = prev_mean_grad + b * num_groups;
  T* m2_grad_ptr = prev_var_grad + b * num_groups;
  T_ACC* du_ptr = du + b * L * num_groups;
  T_ACC* dv_ptr = dv + b * L * num_groups;

  int64_t m0 = count[b];
  T_ACC u_grad = static_cast<T_ACC>(mean_grad_ptr[g]);
  T_ACC v_grad = static_cast<T_ACC>(var_grad_ptr[g]);

  // TODO: Improve this.
  for (int64_t i = L - 1; i >= 0; --i) {
    const T_ACC prev_u = i == 0
                             ? static_cast<T_ACC>(prev_mean[b * num_groups + g])
                             : mean_ptr[(i - 1) * num_groups + g];
    const T_ACC ux = group_mean_ptr[i * num_groups + g];
    const T_ACC u = mean_ptr[i * num_groups + g];
    const T_ACC r = rstd_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const T_ACC c1 = static_cast<T_ACC>(m0 - 1) / static_cast<T_ACC>(m0);
    const T_ACC c2 = T_ACC(1) / static_cast<T_ACC>(m0);

    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g];
    const T_ACC dv = v_grad - T_ACC(0.5) * cuda_utils::Cube<T_ACC>(r) *
                                  ds_ptr[i * num_groups + g];
    du_ptr[i * num_groups + g] =
        c2 * du + T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    dv_ptr[i * num_groups + g] = c2 * dv;
    u_grad = mask ? u_grad : c1 * du - T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    v_grad = mask ? v_grad : c1 * dv;
    m0 -= mask ? 0 : 1;
  }

  m1_grad_ptr[g] = static_cast<T>(u_grad);
  m2_grad_ptr[g] = static_cast<T>(v_grad);
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsLargeKernel(
    int64_t L, int64_t num_groups, int64_t chunk_size,
    const T* __restrict__ mean_grad, const T* __restrict__ var_grad,
    const T* __restrict__ prev_mean, const T_ACC* __restrict__ group_mean,
    const int64_t* __restrict__ count, const T_ACC* __restrict__ cummean,
    const T_ACC* __restrict__ cumrstd, const bool* __restrict__ padding_mask,
    const T_ACC* ds, const T_ACC* db, T* __restrict__ prev_mean_grad,
    T* __restrict__ prev_var_grad, T_ACC* du, T_ACC* dv) {
  __shared__ int64_t
      m0_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m1_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC du_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC dv_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (g >= num_groups) {
    return;
  }

  const T* mean_grad_ptr = mean_grad + b * num_groups;
  const T* var_grad_ptr = var_grad + b * num_groups;
  const T_ACC* group_mean_ptr = group_mean + b * L * num_groups;
  const T_ACC* mean_ptr = cummean + b * L * num_groups;
  const T_ACC* rstd_ptr = cumrstd + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC* ds_ptr = ds + b * L * num_groups;
  const T_ACC* db_ptr = db + b * L * num_groups;

  T* m1_grad_ptr = prev_mean_grad + b * num_groups;
  T* m2_grad_ptr = prev_var_grad + b * num_groups;
  T_ACC* du_ptr = du + b * L * num_groups;
  T_ACC* dv_ptr = dv + b * L * num_groups;

  int64_t cnt = 0;
  utils::KahanWrapper<T_ACC> sum(T_ACC(0));
  for (int64_t i = l; i < r; ++i) {
    const T_ACC ux = group_mean_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    cnt += mask ? 0 : 1;
    // sum += mask ? T_ACC(0) : ux;
    const utils::KahanWrapper<T_ACC> nxt = sum + ux;
    sum = mask ? sum : nxt;
  }
  m0_shared[threadIdx.y][threadIdx.x] = cnt;
  m1_shared[threadIdx.y][threadIdx.x] =
      cnt == 0 ? T_ACC(0) : *sum / static_cast<T_ACC>(cnt);
  __syncthreads();

  if (threadIdx.y == 0) {
    int64_t m0 = count[b];
#pragma unroll
    for (int64_t i = cuda_utils::kWarpSize / 2 - 1; i >= 0; --i) {
      const int64_t cur = m0_shared[i][threadIdx.x];
      m0_shared[i][threadIdx.x] = m0;
      m0 -= cur;
    }
  }
  __syncthreads();

  int64_t m0 = m0_shared[threadIdx.y][threadIdx.x];
  T_ACC u_grad = T_ACC(0);
  T_ACC v_grad = T_ACC(0);
  for (int64_t i = r - 1; i >= l; --i) {
    const T_ACC prev_u = i == 0
                             ? static_cast<T_ACC>(prev_mean[b * num_groups + g])
                             : mean_ptr[(i - 1) * num_groups + g];
    const T_ACC ux = group_mean_ptr[i * num_groups + g];
    const T_ACC u = mean_ptr[i * num_groups + g];
    const T_ACC r = rstd_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const T_ACC c1 = static_cast<T_ACC>(m0 - 1) / static_cast<T_ACC>(m0);
    const T_ACC c2 = T_ACC(1) / static_cast<T_ACC>(m0);

    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g];
    const T_ACC dv = v_grad - T_ACC(0.5) * cuda_utils::Cube<T_ACC>(r) *
                                  ds_ptr[i * num_groups + g];
    u_grad = mask ? u_grad : c1 * du - T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    v_grad = mask ? v_grad : c1 * dv;
    m0 -= mask ? 0 : 1;
  }
  du_shared[threadIdx.y][threadIdx.x] = u_grad;
  dv_shared[threadIdx.y][threadIdx.x] = v_grad;
  __syncthreads();

  if (threadIdx.y == 0) {
    u_grad = static_cast<T_ACC>(mean_grad_ptr[g]);
    v_grad = static_cast<T_ACC>(var_grad_ptr[g]);
#pragma unroll
    for (int64_t i = cuda_utils::kWarpSize / 2 - 1; i >= 0; --i) {
      const T_ACC prev_u =
          i == 0 ? static_cast<T_ACC>(prev_mean[b * num_groups + g])
                 : mean_ptr[(i * chunk_size - 1) * num_groups + g];
      const int64_t n = m0_shared[i][threadIdx.x];
      const int64_t m = i == 0 ? n - cnt : m0_shared[i - 1][threadIdx.x];
      const T_ACC c1 = static_cast<T_ACC>(m) / static_cast<T_ACC>(n);
      const T_ACC c2 = T_ACC(1) - c1;
      const T_ACC m1x = m1_shared[i][threadIdx.x];
      const T_ACC dux = du_shared[i][threadIdx.x];
      const T_ACC dvx = dv_shared[i][threadIdx.x];
      du_shared[i][threadIdx.x] = u_grad;
      dv_shared[i][threadIdx.x] = v_grad;
      u_grad = dux + c1 * u_grad - T_ACC(2) * c1 * c2 * v_grad * (m1x - prev_u);
      v_grad = dvx + c1 * v_grad;
    }
  }
  __syncthreads();

  m0 = m0_shared[threadIdx.y][threadIdx.x];
  u_grad = du_shared[threadIdx.y][threadIdx.x];
  v_grad = dv_shared[threadIdx.y][threadIdx.x];
  for (int64_t i = r - 1; i >= l; --i) {
    const T_ACC prev_u = i == 0
                             ? static_cast<T_ACC>(prev_mean[b * num_groups + g])
                             : mean_ptr[(i - 1) * num_groups + g];
    const T_ACC ux = group_mean_ptr[i * num_groups + g];
    const T_ACC u = mean_ptr[i * num_groups + g];
    const T_ACC r = rstd_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const T_ACC c1 = static_cast<T_ACC>(m0 - 1) / static_cast<T_ACC>(m0);
    const T_ACC c2 = T_ACC(1) / static_cast<T_ACC>(m0);

    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g];
    const T_ACC dv = v_grad - T_ACC(0.5) * cuda_utils::Cube<T_ACC>(r) *
                                  ds_ptr[i * num_groups + g];
    du_ptr[i * num_groups + g] =
        c2 * du + T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    dv_ptr[i * num_groups + g] = c2 * dv;
    u_grad = mask ? u_grad : c1 * du - T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    v_grad = mask ? v_grad : c1 * dv;
    m0 -= mask ? 0 : 1;
  }
  if (threadIdx.y == 0) {
    m1_grad_ptr[g] = static_cast<T>(u_grad);
    m2_grad_ptr[g] = static_cast<T>(v_grad);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupTimestepNormCUDABwdKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* __restrict__ y_grad,
    const T* __restrict__ x, const T_ACC* __restrict__ group_mean,
    const T_ACC* __restrict__ cumrstd, const T* __restrict__ gamma,
    const bool* __restrict__ padding_mask, const T_ACC* __restrict__ du,
    const T_ACC* __restrict__ dv, T* __restrict__ x_grad) {
  extern __shared__ float shm[];

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t l = blockIdx.x;
  const T_ACC coef = T_ACC(1) / T_ACC(D);

  T_ACC* u_shared = reinterpret_cast<T_ACC*>(shm);
  T_ACC* r_shared = u_shared + num_groups;
  T_ACC* du_shared = r_shared + num_groups;
  T_ACC* dv_shared = du_shared + num_groups;

  const T* dy_ptr = y_grad + (b * L + l) * H;
  const T* x_ptr = x + (b * L + l) * H;
  const T_ACC* group_mean_ptr = group_mean + (b * L + l) * num_groups;
  const T_ACC* cumrstd_ptr = cumrstd + (b * L + l) * num_groups;
  const bool mask = padding_mask != nullptr && padding_mask[b * L + l];
  const T_ACC* du_ptr = du + (b * L + l) * num_groups;
  const T_ACC* dv_ptr = dv + (b * L + l) * num_groups;
  T* dx_ptr = x_grad + (b * L + l) * H;

  if (mask) {
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
      dx_ptr[i] = T(0);
    }
    return;
  }

  for (int64_t i = threadIdx.x; i < num_groups; i += blockDim.x) {
    u_shared[i] = group_mean_ptr[i];
    r_shared[i] = cumrstd_ptr[i];
    du_shared[i] = du_ptr[i];
    dv_shared[i] = dv_ptr[i];
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t g = i / D;
    const T_ACC dy_acc = static_cast<T_ACC>(dy_ptr[i]);
    const T_ACC x_acc = static_cast<T_ACC>(x_ptr[i]);
    const T_ACC ux = u_shared[g];
    const T_ACC r = r_shared[g];
    const T_ACC w_acc = static_cast<T_ACC>(gamma[i]);
    const T_ACC dux = du_shared[g];
    const T_ACC dvx = dv_shared[g];
    dx_ptr[i] = static_cast<T>(dy_acc * r * w_acc +
                               coef * (dux + T_ACC(2) * dvx * (x_acc - ux)));
  }
}

template <typename T, typename T_ACC>
__global__ void GroupGammaBetaCUDABwdSmallKernel(
    int64_t outer_size, int64_t inner_size, int64_t num_groups,
    const T* __restrict__ y_grad, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, T* __restrict__ w_grad,
    T* __restrict__ b_grad) {
  const int64_t D = inner_size / num_groups;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = j / D;
  if (j >= inner_size) {
    return;
  }
  utils::KahanWrapper<T_ACC> sum1(T_ACC(0));
  utils::KahanWrapper<T_ACC> sum2(T_ACC(0));
  for (int64_t i = 0; i < outer_size; ++i) {
    const bool mask = padding_mask != nullptr && padding_mask[i];
    const T_ACC dy_acc = static_cast<T_ACC>(y_grad[i * inner_size + j]);
    const T_ACC x_acc = static_cast<T_ACC>(x[i * inner_size + j]);
    const T_ACC u = cummean[i * num_groups + g];
    const T_ACC r = cumrstd[i * num_groups + g];
    // sum1 += mask ? T_ACC(0) : dy_acc * (x_acc - u) * r;
    // sum2 += mask ? T_ACC(0) : dy_acc;
    const utils::KahanWrapper<T_ACC> t1 = sum1 + dy_acc * (x_acc - u) * r;
    const utils::KahanWrapper<T_ACC> t2 = sum2 + dy_acc;
    sum1 = mask ? sum1 : t1;
    sum2 = mask ? sum2 : t2;
  }
  w_grad[j] = static_cast<T>(*sum1);
  b_grad[j] = static_cast<T>(*sum2);
}

template <typename T, typename T_ACC>
__global__ void GroupGammaBetaCUDABwdLargeKernel(
    int64_t outer_size, int64_t inner_size, int64_t num_groups,
    const T* __restrict__ y_grad, const T* __restrict__ x,
    const T_ACC* __restrict__ cummean, const T_ACC* __restrict__ cumrstd,
    const bool* __restrict__ padding_mask, T* __restrict__ w_grad,
    T* __restrict__ b_grad) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t D = inner_size / num_groups;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = j / D;

  utils::KahanWrapper<T_ACC> sum1(T_ACC(0));
  utils::KahanWrapper<T_ACC> sum2(T_ACC(0));
  for (int64_t i = threadIdx.y; i < outer_size; i += blockDim.y) {
    const bool mask = padding_mask != nullptr && padding_mask[i];
    const T_ACC dy_acc = j < inner_size
                             ? static_cast<T_ACC>(y_grad[i * inner_size + j])
                             : T_ACC(0);
    const T_ACC x_acc =
        j < inner_size ? static_cast<T_ACC>(x[i * inner_size + j]) : T_ACC(0);
    const T_ACC u = g < num_groups ? cummean[i * num_groups + g] : T_ACC(0);
    const T_ACC r = g < num_groups ? cumrstd[i * num_groups + g] : T_ACC(0);
    // sum1 += mask ? T_ACC(0) : dy_acc * (x_acc - u) * r;
    // sum2 += mask ? T_ACC(0) : dy_acc;
    const utils::KahanWrapper<T_ACC> t1 = sum1 + dy_acc * (x_acc - u) * r;
    const utils::KahanWrapper<T_ACC> t2 = sum2 + dy_acc;
    sum1 = mask ? sum1 : t1;
    sum2 = mask ? sum2 : t2;
  }
  ds_shared[threadIdx.x][threadIdx.y] = *sum1;
  db_shared[threadIdx.x][threadIdx.y] = *sum2;
  __syncthreads();

  T_ACC s1 = ds_shared[threadIdx.y][threadIdx.x];
  T_ACC s2 = db_shared[threadIdx.y][threadIdx.x];
  s1 = reduce::WarpReduce(s1);
  s2 = reduce::WarpReduce(s2);

  if (threadIdx.x == 0) {
    const int64_t h = blockIdx.x * blockDim.x + threadIdx.y;
    if (h < inner_size) {
      w_grad[h] = static_cast<T>(s1);
      b_grad[h] = static_cast<T>(s2);
    }
  }
}

#define DISPATCH_ROWWISE_REDUCE_CUDA_KERNEL(                                  \
    KernelFunc, T, T_ACC, outer_size, inner_size, shm_size, cuda_stream, ...) \
  do {                                                                        \
    if (inner_size <= 32) {                                                   \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 32, 32>, outer_size, 32,  \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 64) {                                            \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 64, 32>, outer_size, 32,  \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 128) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 128, 32>, outer_size, 32, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 256) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32>, outer_size, 32, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 512) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64>, outer_size, 64, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 1024) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128>, outer_size,   \
                               128, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 2048) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256>, outer_size,   \
                               256, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 4096) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512>, outer_size,   \
                               512, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 8192) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8192, 1024>, outer_size,  \
                               1024, shm_size, cuda_stream, __VA_ARGS__);     \
    } else {                                                                  \
      TORCH_CHECK(false);                                                     \
    }                                                                         \
  } while (false)

template <typename T>
void TimestepNormCUDAFwdImpl(
    const torch::Tensor& x, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, double eps, torch::Tensor& y,
    torch::Tensor& count, torch::Tensor& mean, torch::Tensor& var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);

  const T* x_data = x.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* y_data = y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T* cummean_data = cummean.data_ptr<T>();
  T* cumrstd_data = cumrstd.data_ptr<T>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
    TimestepNormCUDAFwdSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, H, x_data, prev_count_data, prev_mean_data, prev_var_data,
            gamma_data, beta_data, padding_mask_data, static_cast<T_ACC>(eps),
            y_data, count_data, mean_data, var_data, cummean_data,
            cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(H, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp(L, cuda_utils::kWarpSize);
    TimestepNormCUDAFwdLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, H, chunk_size, x_data, prev_count_data,
                          prev_mean_data, prev_var_data, gamma_data, beta_data,
                          padding_mask_data, static_cast<T_ACC>(eps), y_data,
                          count_data, mean_data, var_data, cummean_data,
                          cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T>
void TimestepNormCUDABwdImpl(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& padding_mask,
    torch::Tensor& x_grad, torch::Tensor& prev_mean_grad,
    torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);

  torch::Tensor w_grad = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor b_grad = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* x_data = x.data_ptr<T>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T* cummean_data = cummean.data_ptr<T>();
  const T* cumrstd_data = cumrstd.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* x_grad_data = x_grad.data_ptr<T>();
  T* prev_mean_grad_data = prev_mean_grad.data_ptr<T>();
  T* prev_var_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* w_grad_data = w_grad.data_ptr<T_ACC>();
  T_ACC* b_grad_data = b_grad.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
    TimestepNormCUDABwdSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, H, y_grad_data, mean_grad_data, var_grad_data, x_data,
            prev_mean_data, count_data, cummean_data, cumrstd_data, gamma_data,
            padding_mask_data, x_grad_data, prev_mean_grad_data,
            prev_var_grad_data, w_grad_data, b_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(H, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp(L, cuda_utils::kWarpSize / 2);
    TimestepNormCUDABwdLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize / 2),
           0, cuda_stream>>>(
            L, H, chunk_size, y_grad_data, mean_grad_data, var_grad_data,
            x_data, prev_mean_data, count_data, cummean_data, cumrstd_data,
            gamma_data, padding_mask_data, x_grad_data, prev_mean_grad_data,
            prev_var_grad_data, w_grad_data, b_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, w_grad_data, b_grad_data, gamma_grad_data, beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupTimestepNormCUDAFwdImpl(
    const torch::Tensor& x, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, int64_t num_groups, double eps,
    torch::Tensor& y, torch::Tensor& count, torch::Tensor& mean,
    torch::Tensor& var, torch::Tensor& group_mean, torch::Tensor& group_var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);
  const int64_t D = H / num_groups;

  const T* x_data = x.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* y_data = y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();
  T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    constexpr int64_t kShmSize = 0;
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(D);
    DISPATCH_ROWWISE_REDUCE_CUDA_KERNEL(
        RowwiseMomentsKernel, T, T_ACC, B * L * num_groups, D, kShmSize,
        cuda_stream, D, x_data, group_mean_data, group_var_data);
  }
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads = (num_groups < cuda_utils::kCUDANumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDANumThreads);
    const int64_t M = utils::DivUp(num_groups, num_threads);
    ColwiseCumMomentsSmallKernel<T, T_ACC>
        <<<dim3(M, B), num_threads, 0, cuda_stream>>>(
            L, num_groups, prev_count_data, prev_mean_data, prev_var_data,
            group_mean_data, group_var_data, padding_mask_data,
            static_cast<T_ACC>(eps), count_data, mean_data, var_data,
            cummean_data, cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(num_groups, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp(L, cuda_utils::kWarpSize);
    ColwiseCumMomentsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, num_groups, chunk_size, prev_count_data,
                          prev_mean_data, prev_var_data, group_mean_data,
                          group_var_data, padding_mask_data,
                          static_cast<T_ACC>(eps), count_data, mean_data,
                          var_data, cummean_data, cumrstd_data);
  }
  {
    const int64_t shm_size = sizeof(T_ACC) * num_groups * 2;
    cuda_utils::LaunchKernel(GroupTimestepNormCUDAFwdKernel<T, T_ACC>,
                             dim3(L, B), cuda_utils::kCUDANumThreads, shm_size,
                             cuda_stream, L, H, num_groups, x_data,
                             cummean_data, cumrstd_data, gamma_data, beta_data,
                             padding_mask_data, y_data);
  }
}

template <typename T>
void GroupTimestepNormCUDABwdImpl(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& padding_mask,
    int64_t num_groups, torch::Tensor& x_grad, torch::Tensor& prev_mean_grad,
    torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t H = x.size(2);
  const int64_t D = H / num_groups;

  torch::Tensor group_mean =
      torch::empty({B, L, num_groups},
                   x.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor ds =
      torch::empty({B, L, num_groups},
                   x.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db =
      torch::empty({B, L, num_groups},
                   x.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* x_data = x.data_ptr<T>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  const T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* x_grad_data = x_grad.data_ptr<T>();
  T* prev_mean_grad_data = prev_mean_grad.data_ptr<T>();
  T* prev_var_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    const int64_t block_size =
        std::max(int64_t(1) << utils::CeilLog2(D), cuda_utils::kWarpSize);
    const int64_t shm_size = sizeof(T_ACC) * cuda_utils::kWarpSize * 3 +
                             sizeof(T_ACC) * block_size * 2;
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(D);
    DISPATCH_ROWWISE_REDUCE_CUDA_KERNEL(
        RowwiseInternalGradientsKernel, T, T_ACC, dim3(B * L, num_groups), D,
        shm_size, cuda_stream, num_groups, D, y_grad_data, x_data, cummean_data,
        gamma_data, group_mean_data, ds_data, db_data);
  }
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(num_groups);
    const int64_t M = utils::DivUp<int64_t>(num_groups, num_threads);
    ColwiseInternalGradientsSmallKernel<T, T_ACC>
        <<<dim3(M, B), num_threads, 0, cuda_stream>>>(
            L, num_groups, mean_grad_data, var_grad_data, prev_mean_data,
            group_mean_data, count_data, cummean_data, cumrstd_data,
            padding_mask_data, ds_data, db_data, prev_mean_grad_data,
            prev_var_grad_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp<int64_t>(num_groups, cuda_utils::kWarpSize);
    const int64_t chunk_size =
        utils::DivUp<int64_t>(L, cuda_utils::kWarpSize / 2);
    ColwiseInternalGradientsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize / 2),
           0, cuda_stream>>>(
            L, num_groups, chunk_size, mean_grad_data, var_grad_data,
            prev_mean_data, group_mean_data, count_data, cummean_data,
            cumrstd_data, padding_mask_data, ds_data, db_data,
            prev_mean_grad_data, prev_var_grad_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  {
    const int64_t shm_size = sizeof(T_ACC) * num_groups * 4;
    cuda_utils::LaunchKernel(GroupTimestepNormCUDABwdKernel<T, T_ACC>,
                             dim3(L, B), cuda_utils::kCUDANumThreads, shm_size,
                             cuda_stream, L, H, num_groups, y_grad_data, x_data,
                             group_mean_data, cumrstd_data, gamma_data,
                             padding_mask_data, ds_data, db_data, x_grad_data);
  }

  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads = cuda_utils::RowwiseNumThreads(num_groups);
    const int64_t M = utils::DivUp<int64_t>(H, num_threads);
    GroupGammaBetaCUDABwdSmallKernel<T, T_ACC>
        <<<M, num_threads, 0, cuda_stream>>>(
            B * L, H, num_groups, y_grad_data, x_data, cummean_data,
            cumrstd_data, padding_mask_data, gamma_grad_data, beta_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp<int64_t>(H, cuda_utils::kWarpSize);
    GroupGammaBetaCUDABwdLargeKernel<T, T_ACC>
        <<<M, dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(B * L, H, num_groups, y_grad_data, x_data,
                          cummean_data, cumrstd_data, padding_mask_data,
                          gamma_grad_data, beta_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

#undef DISPATCH_ROWWISE_REDUCE_CUDA_KERNEL

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCUDAFwd(const torch::Tensor& x, const torch::Tensor& prev_count,
                    const torch::Tensor& prev_mean,
                    const torch::Tensor& prev_var, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps) {
  const int64_t B = x.size(0);
  const int64_t L = x.size(1);
  const int64_t N = x.size(2);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor y = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count,
      prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(
      prev_mean,
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty(
      {B, L, N},
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty(
      {B, L, N},
      prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "TimestepNormCUDAFwd", [&]() {
        TimestepNormCUDAFwdImpl<scalar_t>(
            *(x.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), eps, y, count,
            mean, var, cummean, cumrstd);
      });
  return std::make_tuple(y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
                    const torch::Tensor& var_grad, const torch::Tensor& x,
                    const torch::Tensor& prev_mean, const torch::Tensor& count,
                    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& padding_mask) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor x_grad = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
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
      at::kHalf, at::kBFloat16, x.scalar_type(), "TimestepNormCUDABwd", [&]() {
        TimestepNormCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(count.expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), x_grad,
            prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(x_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
GroupTimestepNormCUDAFwd(const torch::Tensor& x,
                         const torch::Tensor& prev_count,
                         const torch::Tensor& prev_mean,
                         const torch::Tensor& prev_var,
                         const torch::Tensor& gamma, const torch::Tensor& beta,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, double eps) {
  const int64_t B = x.size(0);
  const int64_t L = x.size(1);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor y = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count,
      prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(
      prev_mean,
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(x.scalar_type(), true);
  torch::Tensor group_mean = torch::empty(
      {B, L, num_groups},
      x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor group_var = torch::empty(
      {B, L, num_groups},
      x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty(
      {B, L, num_groups},
      x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty(
      {B, L, num_groups},
      x.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupTimestepNormCUDAFwd",
      [&]() {
        GroupTimestepNormCUDAFwdImpl<scalar_t>(
            *(x.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups, eps,
            y, count, mean, var, group_mean, group_var, cummean, cumrstd);
      });

  return std::make_tuple(y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& x,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor x_grad = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
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
      at::kHalf, at::kBFloat16, x.scalar_type(), "GroupTimestepNormCUDABwd",
      [&]() {
        GroupTimestepNormCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(count.expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups,
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(x_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

}  // namespace ops
}  // namespace megalodon
