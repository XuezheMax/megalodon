#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <thrust/tuple.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include "cuda_utils.cuh"
#include "ops/sequence_norm.h"
#include "reduce.cuh"
#include "welford.h"

namespace megalodon {
namespace ops {

namespace {

template <typename T, typename T_ACC>
__global__ void RowwiseMomentsKernel(int64_t H, int64_t L, const T* X,
                                     const bool* padding_mask, T_ACC eps,
                                     int64_t* count, T_ACC* mean, T_ACC* rstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData shm[cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x;
  const T* X_ptr = X + (b * H + h) * L;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;

  utils::WelfordData<T_ACC> m;
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    m = mask ? m : m + x;
    // const utils::WelfordData<T_ACC> nxt = m + x;
    // m = mask ? m : nxt;
  }
  m = reduce::BlockReduce(m, shm_ptr);

  if (threadIdx.x == 0) {
    if (h == 0) {
      count[b] = m.m0;
    }
    mean[b * H + h] = m.m1;
    rstd[b * H + h] = c10::cuda::compat::rsqrt(m.m2 + eps);
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseMomentsSmallKernel(int64_t L, int64_t H, const T* X,
                                          const bool* padding_mask, T_ACC eps,
                                          int64_t* count, T_ACC* mean,
                                          T_ACC* rstd) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* X_ptr = X + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* mean_ptr = mean + i * H;
  T_ACC* rstd_ptr = rstd + i * H;

  utils::WelfordData<T_ACC> m;
  for (int64_t j = 0; j < L; ++j) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const utils::WelfordData<T_ACC> nxt = m + x;
    m = mask ? m : nxt;
  }
  if (k == 0) {
    count[i] = m.m0;
  }
  mean_ptr[k] = m.m1;
  rstd_ptr[k] = c10::cuda::compat::rsqrt(m.m2 + eps);
}

template <typename T, typename T_ACC>
__global__ void ColwiseMomentsLargeKernel(int64_t L, int64_t H, const T* X,
                                          const bool* padding_mask, T_ACC eps,
                                          int64_t* count, T_ACC* mean,
                                          T_ACC* rstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData
      shm[cuda_utils::kWarpSize * cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* X_ptr = X + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* mean_ptr = mean + i * H;
  T_ACC* rstd_ptr = rstd + i * H;

  utils::WelfordData<T_ACC> m;
  for (int64_t j = threadIdx.y; j < L; j += blockDim.y) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const utils::WelfordData<T_ACC> nxt = m + x;
    m = mask ? m : nxt;
  }
  shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x] = m;
  __syncthreads();

#pragma unroll
  for (int64_t offset = cuda_utils::kWarpSize >> 1; offset >= 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      shm_ptr[threadIdx.y * cuda_utils::kWarpSize + threadIdx.x] +=
          shm_ptr[(threadIdx.y + offset) * cuda_utils::kWarpSize + threadIdx.x];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0) {
    if (k == 0) {
      count[i] = shm_ptr[0].m0;
    }
    mean_ptr[k] = shm_ptr[threadIdx.x].m1;
    rstd_ptr[k] = c10::cuda::compat::rsqrt(shm_ptr[threadIdx.x].m2 + eps);
  }
}

template <typename T, typename T_ACC>
__global__ void SequenceNormCUDAFwdBLHKernel(int64_t L, int64_t H, const T* X,
                                             const T_ACC* mean,
                                             const T_ACC* rstd, const T* gamma,
                                             const T* beta,
                                             const bool* padding_mask, T* Y) {
  const int64_t i = blockIdx.y;
  const int64_t j = blockIdx.x;
  const T* X_ptr = X + (i * L + j) * H;
  const T_ACC* mean_ptr = mean + i * H;
  const T_ACC* rstd_ptr = rstd + i * H;
  T* Y_ptr = Y + (i * L + j) * H;
  const bool mask = padding_mask != nullptr && padding_mask[i * L + j];
  if (mask) {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      Y_ptr[k] = T(0);
    }
  } else {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      const T_ACC x = static_cast<T_ACC>(X_ptr[k]);
      const T_ACC w = static_cast<T_ACC>(gamma[k]);
      const T_ACC b = static_cast<T_ACC>(beta[k]);
      Y_ptr[k] = static_cast<T>((x - mean_ptr[k]) * rstd_ptr[k] * w + b);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void SequenceNormCUDAFwdBHLKernel(int64_t H, int64_t L, const T* X,
                                             const T_ACC* mean,
                                             const T_ACC* rstd, const T* gamma,
                                             const T* beta,
                                             const bool* padding_mask, T* Y) {
  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x;
  const T* X_ptr = X + (b * H + h) * L;
  const T_ACC u = mean[b * H + h];
  const T_ACC r = rstd[b * H + h];
  const T_ACC weight = static_cast<T_ACC>(gamma[h]);
  const T_ACC bias = static_cast<T_ACC>(beta[h]);
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T* Y_ptr = Y + (b * H + h) * L;
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    Y_ptr[i] = mask ? T(0) : static_cast<T>((x - u) * r * weight + bias);
  }
}

template <typename T, typename T_ACC>
__global__ void RowwiseInternalGradientsKernel(int64_t H, int64_t L,
                                               const T* Y_grad, const T* X,
                                               const bool* padding_mask,
                                               T_ACC* ds, T_ACC* db) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize];

  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x;
  const T* Y_grad_ptr = Y_grad + (b * H + h) * L;
  const T* X_ptr = X + (b * H + h) * L;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;

  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[i]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    sum1 += mask ? T_ACC(0) : dy * x;
    sum2 += mask ? T_ACC(0) : dy;
  }
  sum1 = reduce::BlockReduce(sum1, ds_shared);
  sum2 = reduce::BlockReduce(sum2, db_shared);
  if (threadIdx.x == 0) {
    ds[b * H + h] = sum1;
    db[b * H + h] = sum2;
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsSmallKernel(int64_t L, int64_t H,
                                                    const T* Y_grad, const T* X,
                                                    const bool* padding_mask,
                                                    T_ACC* ds, T_ACC* db) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + i * L * H;
  const T* X_ptr = X + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* ds_ptr = ds + i * H;
  T_ACC* db_ptr = db + i * H;

  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t j = 0; j < L; ++j) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + k]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    sum1 += mask ? T_ACC(0) : dy * x;
    sum2 += mask ? T_ACC(0) : dy;
  }
  ds_ptr[k] = sum1;
  db_ptr[k] = sum2;
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsLargeKernel(int64_t L, int64_t H,
                                                    const T* Y_grad, const T* X,
                                                    const bool* padding_mask,
                                                    T_ACC* ds, T_ACC* db) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + i * L * H;
  const T* X_ptr = X + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* ds_ptr = ds + i * H;
  T_ACC* db_ptr = db + i * H;

  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t j = threadIdx.y; j < L; j += blockDim.y) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + k]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    sum1 += mask ? T_ACC(0) : dy * x;
    sum2 += mask ? T_ACC(0) : dy;
  }
  ds_shared[threadIdx.y][threadIdx.x] = sum1;
  db_shared[threadIdx.y][threadIdx.x] = sum2;
  __syncthreads();

#pragma unroll
  for (int64_t offset = cuda_utils::kWarpSize >> 1; offset >= 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      ds_shared[threadIdx.y][threadIdx.x] +=
          ds_shared[threadIdx.y + offset][threadIdx.x];
      db_shared[threadIdx.y][threadIdx.x] +=
          db_shared[threadIdx.y + offset][threadIdx.x];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0) {
    ds_ptr[k] = ds_shared[0][threadIdx.x];
    db_ptr[k] = db_shared[0][threadIdx.x];
  }
}

template <typename T, typename T_ACC>
__global__ void SequenceNormCUDABwdBLHKernel(
    int64_t L, int64_t H, const T* Y_grad, const T* X, const int64_t* count,
    const T_ACC* mean, const T_ACC* rstd, const T* gamma,
    const bool* padding_mask, const T_ACC* ds, const T_ACC* db, T* X_grad) {
  const int64_t i = blockIdx.y;
  const int64_t j = blockIdx.x;
  const T* Y_grad_ptr = Y_grad + (i * L + j) * H;
  const T* X_ptr = X + (i * L + j) * H;
  const T_ACC* mean_ptr = mean + i * H;
  const T_ACC* rstd_ptr = rstd + i * H;
  const T_ACC* ds_ptr = ds + i * H;
  const T_ACC* db_ptr = db + i * H;
  T* X_grad_ptr = X_grad + (i * L + j) * H;

  // const int64_t cnt = count[i];
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count[i]);
  const bool mask = padding_mask != nullptr && padding_mask[i * L + j];
  if (mask) {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      X_grad_ptr[k] = T(0);
    }
  } else {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[k]);
      const T_ACC x = static_cast<T_ACC>(X_ptr[k]);
      const T_ACC u = mean_ptr[k];
      const T_ACC r = rstd_ptr[k];
      const T_ACC w = static_cast<T_ACC>(gamma[k]);
      const T_ACC dv =
          T_ACC(0.5) * cuda_utils::Cube(r) * w * (u * db_ptr[k] - ds_ptr[k]);
      const T_ACC du = -r * w * db_ptr[k];
      const T_ACC dx = r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
      X_grad_ptr[k] = static_cast<T>(dx);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void SequenceNormCUDABwdBHLKernel(
    int64_t H, int64_t L, const T* Y_grad, const T* X, const int64_t* count,
    const T_ACC* mean, const T_ACC* rstd, const T* gamma,
    const bool* padding_mask, const T_ACC* ds, const T_ACC* db, T* X_grad) {
  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x;
  const T* Y_grad_ptr = Y_grad + (b * H + h) * L;
  const T* X_ptr = X + (b * H + h) * L;
  // const int64_t cnt = count[i];
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count[b]);
  const T_ACC u = mean[b * H + h];
  const T_ACC r = rstd[b * H + h];
  const T_ACC w = static_cast<T_ACC>(gamma[h]);
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC ds_val = ds[b * H + h];
  const T_ACC db_val = db[b * H + h];
  T* X_grad_ptr = X_grad + (b * H + h) * L;

  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[i]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const T_ACC dv =
        T_ACC(0.5) * cuda_utils::Cube(r) * w * (u * db_val - ds_val);
    const T_ACC du = -r * w * db_val;
    const T_ACC dx = r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
    X_grad_ptr[i] = mask ? T(0) : static_cast<T>(dx);
  }
}

template <typename T, typename T_ACC>
__global__ void GammaBetaCUDABwdKernel(int64_t B, int64_t H, const T_ACC* mean,
                                       const T_ACC* rstd, const T_ACC* ds,
                                       const T_ACC* db, T* gamma_grad,
                                       T* beta_grad) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) {
    return;
  }
  T_ACC w_grad = T_ACC(0);
  T_ACC b_grad = T_ACC(0);
  for (int64_t i = 0; i < B; ++i) {
    const T_ACC u = mean[i * H + j];
    const T_ACC r = rstd[i * H + j];
    w_grad += r * (ds[i * H + j] - u * db[i * H + j]);
    b_grad += db[i * H + j];
  }
  gamma_grad[j] = static_cast<T>(w_grad);
  beta_grad[j] = static_cast<T>(b_grad);
}

template <typename T, typename T_ACC>
__global__ void TransposeRowwiseMomentsKernel(int64_t L, int64_t H,
                                              int64_t num_groups, const T* X,
                                              T_ACC* mean, T_ACC* var) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData shm[cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t k = blockIdx.x;
  const int64_t l = k / num_groups;
  const int64_t g = k % num_groups;
  const T* X_ptr = X + b * L * H + k * D;
  T_ACC* m1_ptr = mean + b * num_groups * L;
  T_ACC* m2_ptr = var + b * num_groups * L;

  utils::WelfordData<T_ACC> m;
  for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    m += x;
  }
  m = reduce::BlockReduce(m, shm_ptr);
  if (threadIdx.x == 0) {
    m1_ptr[g * L + l] = m.m1;
    m2_ptr[g * L + l] = m.m2;
  }
}

template <typename T, typename T_ACC>
__global__ void CombineRowwiseMomentsKernel(int64_t num_groups, int64_t L,
                                            const T_ACC* group_mean,
                                            const T_ACC* group_var,
                                            const bool* padding_mask, T_ACC eps,
                                            int64_t* count, T_ACC* mean,
                                            T_ACC* rstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData shm[cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x;
  const T_ACC* m1_ptr = group_mean + (b * num_groups + g) * L;
  const T_ACC* m2_ptr = group_var + (b * num_groups + g) * L;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;

  utils::WelfordData<T_ACC> m;
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const utils::WelfordData<T_ACC> cur{int64_t(1), m1_ptr[i], m2_ptr[i]};
    m = mask ? m : m + cur;
    // const utils::WelfordData<T_ACC> nxt = m + cur;
    // m = mask ? m : nxt;
  }
  m = reduce::BlockReduce(m, shm_ptr);
  if (threadIdx.x == 0) {
    if (g == 0) {
      count[b] = m.m0;
    }
    mean[b * num_groups + g] = m.m1;
    rstd[b * num_groups + g] = c10::cuda::compat::rsqrt(m.m2 + eps);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupSequenceNormCUDAFwdBLHKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* X, const T_ACC* mean,
    const T_ACC* rstd, const T* gamma, const T* beta, const bool* padding_mask,
    T* Y) {
  const int64_t D = H / num_groups;
  const int64_t i = blockIdx.y;
  const int64_t j = blockIdx.x;
  const T* X_ptr = X + (i * L + j) * H;
  const T_ACC* mean_ptr = mean + i * num_groups;
  const T_ACC* rstd_ptr = rstd + i * num_groups;
  T* Y_ptr = Y + (i * L + j) * H;
  const bool mask = padding_mask != nullptr && padding_mask[i * L + j];
  if (mask) {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      Y_ptr[k] = T(0);
    }
  } else {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      const int64_t g = k / D;
      const T_ACC x = static_cast<T_ACC>(X_ptr[k]);
      const T_ACC w = static_cast<T_ACC>(gamma[k]);
      const T_ACC b = static_cast<T_ACC>(beta[k]);
      Y_ptr[k] = static_cast<T>((x - mean_ptr[g]) * rstd_ptr[g] * w + b);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void GroupRowwiseMomentsKernel(int64_t H, int64_t L,
                                          int64_t num_groups, const T* X,
                                          const bool* padding_mask, T_ACC eps,
                                          int64_t* count, T_ACC* mean,
                                          T_ACC* rstd) {
  using AlignedWelfordData =
      typename std::aligned_storage<sizeof(utils::WelfordData<T_ACC>),
                                    alignof(utils::WelfordData<T_ACC>)>::type;
  __shared__ AlignedWelfordData shm[cuda_utils::kWarpSize];
  utils::WelfordData<T_ACC>* shm_ptr =
      reinterpret_cast<utils::WelfordData<T_ACC>*>(shm);

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x;
  const T* X_ptr = X + (b * num_groups + g) * D * L;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;

  utils::WelfordData<T_ACC> m;
  for (int64_t i = threadIdx.x; i < D * L; i += blockDim.x) {
    const int64_t l = i % L;
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[l];
    m = mask ? m : m + x;
  }
  m = reduce::BlockReduce(m, shm_ptr);
  if (threadIdx.x == 0) {
    if (g == 0) {
      count[b] = m.m0 / D;
    }
    mean[b * num_groups + g] = m.m1;
    rstd[b * num_groups + g] = c10::cuda::compat::rsqrt(m.m2 + eps);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupSequenceNormCUDAFwdBHLKernel(
    int64_t H, int64_t L, int64_t num_groups, const T* X, const T_ACC* mean,
    const T_ACC* rstd, const T* gamma, const T* beta, const bool* padding_mask,
    T* Y) {
  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x;
  const int64_t g = h / D;
  const T* X_ptr = X + (b * H + h) * L;
  const T_ACC u = mean[b * num_groups + g];
  const T_ACC r = rstd[b * num_groups + g];
  const T_ACC weight = static_cast<T_ACC>(gamma[h]);
  const T_ACC bias = static_cast<T_ACC>(beta[h]);
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T* Y_ptr = Y + (b * H + h) * L;
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    Y_ptr[i] = mask ? T(0) : static_cast<T>((x - u) * r * weight + bias);
  }
}

template <typename T, typename T_ACC>
__global__ void CombineRowwiseInternalGradientsKernel(
    int64_t H, int64_t num_groups, const T_ACC* ds, const T_ACC* db,
    const T* gamma, T_ACC* dsw, T_ACC* dbw) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize];

  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x;
  const T_ACC* ds_ptr = ds + b * H + g * D;
  const T_ACC* db_ptr = db + b * H + g * D;
  const T* gamma_ptr = gamma + g * D;

  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
    const T_ACC w = static_cast<T_ACC>(gamma_ptr[i]);
    sum1 += ds_ptr[i] * w;
    sum2 += db_ptr[i] * w;
  }
  sum1 = reduce::BlockReduce(sum1, ds_shared);
  sum2 = reduce::BlockReduce(sum2, db_shared);
  if (threadIdx.x == 0) {
    dsw[b * num_groups + g] = sum1;
    dbw[b * num_groups + g] = sum2;
  }
}

template <typename T, typename T_ACC>
__global__ void GroupSequenceNormCUDABwdBLHKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* Y_grad, const T* X,
    const int64_t* count, const T_ACC* mean, const T_ACC* rstd, const T* gamma,
    const bool* padding_mask, const T_ACC* ds, const T_ACC* db, T* X_grad) {
  const int64_t D = H / num_groups;
  const int64_t i = blockIdx.y;
  const int64_t j = blockIdx.x;

  const T* Y_grad_ptr = Y_grad + (i * L + j) * H;
  const T* X_ptr = X + (i * L + j) * H;
  const T_ACC* mean_ptr = mean + i * num_groups;
  const T_ACC* rstd_ptr = rstd + i * num_groups;
  const T_ACC* ds_ptr = ds + i * num_groups;
  const T_ACC* db_ptr = db + i * num_groups;
  T* X_grad_ptr = X_grad + (i * L + j) * H;

  // const int64_t cnt = count[i];
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count[i] * D);
  const bool mask = padding_mask != nullptr && padding_mask[i * L + j];
  if (mask) {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      X_grad_ptr[k] = T(0);
    }
  } else {
    for (int64_t k = threadIdx.x; k < H; k += blockDim.x) {
      const int64_t g = k / D;
      const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[k]);
      const T_ACC x = static_cast<T_ACC>(X_ptr[k]);
      const T_ACC u = mean_ptr[g];
      const T_ACC r = rstd_ptr[g];
      const T_ACC w = static_cast<T_ACC>(gamma[k]);
      const T_ACC dv =
          T_ACC(0.5) * cuda_utils::Cube(r) * (u * db_ptr[g] - ds_ptr[g]);
      const T_ACC du = -r * db_ptr[g];
      const T_ACC dx = r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
      X_grad_ptr[k] = static_cast<T>(dx);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void GroupSequenceNormCUDABwdBHLKernel(
    int64_t H, int64_t L, int64_t num_groups, const T* Y_grad, const T* X,
    const int64_t* count, const T_ACC* mean, const T_ACC* rstd, const T* gamma,
    const bool* padding_mask, const T_ACC* ds, const T_ACC* db, T* X_grad) {
  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x;
  const int64_t g = h / D;
  const T* Y_grad_ptr = Y_grad + (b * H + h) * L;
  const T* X_ptr = X + (b * H + h) * L;
  // const int64_t cnt = count[i];
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count[b] * D);
  const T_ACC u = mean[b * num_groups + g];
  const T_ACC r = rstd[b * num_groups + g];
  const T_ACC w = static_cast<T_ACC>(gamma[h]);
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC ds_val = ds[b * num_groups + g];
  const T_ACC db_val = db[b * num_groups + g];
  T* X_grad_ptr = X_grad + (b * H + h) * L;

  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[i]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const T_ACC dv = T_ACC(0.5) * cuda_utils::Cube(r) * (u * db_val - ds_val);
    const T_ACC du = -r * db_val;
    const T_ACC dx = r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
    X_grad_ptr[i] = mask ? T(0) : static_cast<T>(dx);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupGammaBetaCUDABwdKernel(int64_t B, int64_t H,
                                            int64_t num_groups,
                                            const T_ACC* mean,
                                            const T_ACC* rstd, const T_ACC* ds,
                                            const T_ACC* db, T* gamma_grad,
                                            T* beta_grad) {
  const int64_t D = H / num_groups;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = j / D;
  if (j >= H) {
    return;
  }
  T_ACC w_grad = T_ACC(0);
  T_ACC b_grad = T_ACC(0);
  for (int64_t i = 0; i < B; ++i) {
    const T_ACC u = mean[i * num_groups + g];
    const T_ACC r = rstd[i * num_groups + g];
    w_grad += r * (ds[i * H + j] - u * db[i * H + j]);
    b_grad += db[i * H + j];
  }
  gamma_grad[j] = static_cast<T>(w_grad);
  beta_grad[j] = static_cast<T>(b_grad);
}

template <typename T>
void SequenceNormCUDAFwdBLHImpl(const torch::Tensor& X,
                                const torch::Tensor& gamma,
                                const torch::Tensor& beta,
                                const torch::Tensor& padding_mask, double eps,
                                torch::Tensor& Y, torch::Tensor& count,
                                torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
    ColwiseMomentsSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, H, X_data, padding_mask_data, static_cast<T_ACC>(eps),
            count_data, mean_data, rstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(H, cuda_utils::kWarpSize);
    ColwiseMomentsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, H, X_data, padding_mask_data,
                          static_cast<T_ACC>(eps), count_data, mean_data,
                          rstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  SequenceNormCUDAFwdBLHKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, X_data, mean_data, rstd_data, gamma_data, beta_data,
          padding_mask_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void SequenceNormCUDAFwdBHLImpl(const torch::Tensor& X,
                                const torch::Tensor& gamma,
                                const torch::Tensor& beta,
                                const torch::Tensor& padding_mask, double eps,
                                torch::Tensor& Y, torch::Tensor& count,
                                torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = (L < cuda_utils::kCUDABlockReduceNumThreads
                                   ? cuda_utils::kWarpSize
                                   : cuda_utils::kCUDABlockReduceNumThreads);
  RowwiseMomentsKernel<T, T_ACC><<<dim3(H, B), num_threads, 0, cuda_stream>>>(
      H, L, X_data, padding_mask_data, static_cast<T_ACC>(eps), count_data,
      mean_data, rstd_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  SequenceNormCUDAFwdBHLKernel<T, T_ACC>
      <<<dim3(H, B), num_threads, 0, cuda_stream>>>(
          H, L, X_data, mean_data, rstd_data, gamma_data, beta_data,
          padding_mask_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void SequenceNormCUDABwdBLHImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, torch::Tensor& X_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);

  torch::Tensor ds = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* mean_data = mean.data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
    ColwiseInternalGradientsSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, H, Y_grad_data, X_data, padding_mask_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(H, cuda_utils::kWarpSize);
    ColwiseInternalGradientsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, H, Y_grad_data, X_data, padding_mask_data, ds_data,
                          db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  SequenceNormCUDABwdBLHKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, Y_grad_data, X_data, count_data, mean_data, rstd_data,
          gamma_data, padding_mask_data, ds_data, db_data, X_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, mean_data, rstd_data, ds_data, db_data, gamma_grad_data,
          beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void SequenceNormCUDABwdBHLImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, torch::Tensor& X_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);

  torch::Tensor ds = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* mean_data = mean.data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = (L < cuda_utils::kCUDABlockReduceNumThreads
                                   ? cuda_utils::kWarpSize
                                   : cuda_utils::kCUDABlockReduceNumThreads);
  RowwiseInternalGradientsKernel<T, T_ACC>
      <<<dim3(H, B), num_threads, 0, cuda_stream>>>(
          H, L, Y_grad_data, X_data, padding_mask_data, ds_data, db_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  SequenceNormCUDABwdBHLKernel<T, T_ACC>
      <<<dim3(H, B), num_threads, 0, cuda_stream>>>(
          H, L, Y_grad_data, X_data, count_data, mean_data, rstd_data,
          gamma_data, padding_mask_data, ds_data, db_data, X_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, mean_data, rstd_data, ds_data, db_data, gamma_grad_data,
          beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupSequenceNormCUDAFwdBLHImpl(const torch::Tensor& X,
                                     const torch::Tensor& gamma,
                                     const torch::Tensor& beta,
                                     const torch::Tensor& padding_mask,
                                     int64_t num_groups, double eps,
                                     torch::Tensor& Y, torch::Tensor& count,
                                     torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t D = H / num_groups;

  const auto acc_type = at::toOpMathType(X.scalar_type());
  torch::Tensor group_mean = torch::empty(
      {B, num_groups, L},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor group_var = torch::empty(
      {B, num_groups, L},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    const int64_t num_threads = (D < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    TransposeRowwiseMomentsKernel<T, T_ACC>
        <<<dim3(L * num_groups, B), num_threads, 0, cuda_stream>>>(
            L, H, num_groups, X_data, group_mean_data, group_var_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    const int64_t num_threads = (L < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    CombineRowwiseMomentsKernel<T, T_ACC>
        <<<dim3(num_groups, B), num_threads, 0, cuda_stream>>>(
            num_groups, L, group_mean_data, group_var_data, padding_mask_data,
            static_cast<T_ACC>(eps), count_data, mean_data, rstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  GroupSequenceNormCUDAFwdBLHKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, num_groups, X_data, mean_data, rstd_data, gamma_data, beta_data,
          padding_mask_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupSequenceNormCUDAFwdBHLImpl(const torch::Tensor& X,
                                     const torch::Tensor& gamma,
                                     const torch::Tensor& beta,
                                     const torch::Tensor& padding_mask,
                                     int64_t num_groups, double eps,
                                     torch::Tensor& Y, torch::Tensor& count,
                                     torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);
  const int64_t D = H / num_groups;

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = (D * L < cuda_utils::kCUDABlockReduceNumThreads
                                   ? cuda_utils::kWarpSize
                                   : cuda_utils::kCUDABlockReduceNumThreads);
  GroupRowwiseMomentsKernel<T, T_ACC>
      <<<dim3(num_groups, B), num_threads, 0, cuda_stream>>>(
          H, L, num_groups, X_data, padding_mask_data, static_cast<T_ACC>(eps),
          count_data, mean_data, rstd_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GroupSequenceNormCUDAFwdBHLKernel<T, T_ACC>
      <<<dim3(H, B), num_threads, 0, cuda_stream>>>(
          H, L, num_groups, X_data, mean_data, rstd_data, gamma_data, beta_data,
          padding_mask_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupSequenceNormCUDABwdBLHImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, int64_t num_groups,
    torch::Tensor& X_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t D = H / num_groups;

  torch::Tensor ds = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor dsw = torch::empty(
      {B, num_groups},
      gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor dbw = torch::empty(
      {B, num_groups},
      gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* mean_data = mean.data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  T_ACC* dsw_data = dsw.data_ptr<T_ACC>();
  T_ACC* dbw_data = dbw.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
    ColwiseInternalGradientsSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, H, Y_grad_data, X_data, padding_mask_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(H, cuda_utils::kWarpSize);
    ColwiseInternalGradientsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, H, Y_grad_data, X_data, padding_mask_data, ds_data,
                          db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t num_threads = (D < cuda_utils::kCUDABlockReduceNumThreads
                                   ? cuda_utils::kWarpSize
                                   : cuda_utils::kCUDABlockReduceNumThreads);
  CombineRowwiseInternalGradientsKernel<T, T_ACC>
      <<<dim3(num_groups, B), num_threads, 0, cuda_stream>>>(
          H, num_groups, ds_data, db_data, gamma_data, dsw_data, dbw_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  GroupSequenceNormCUDABwdBLHKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, num_groups, Y_grad_data, X_data, count_data, mean_data,
          rstd_data, gamma_data, padding_mask_data, dsw_data, dbw_data,
          X_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
  GroupGammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, num_groups, mean_data, rstd_data, ds_data, db_data,
          gamma_grad_data, beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupSequenceNormCUDABwdBHLImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, int64_t num_groups,
    torch::Tensor& X_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);
  const int64_t D = H / num_groups;

  torch::Tensor ds = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor dsw = torch::empty(
      {B, num_groups},
      gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor dbw = torch::empty(
      {B, num_groups},
      gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* mean_data = mean.data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  T_ACC* dsw_data = dsw.data_ptr<T_ACC>();
  T_ACC* dbw_data = dbw.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    const int64_t num_threads = (L < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    RowwiseInternalGradientsKernel<T, T_ACC>
        <<<dim3(H, B), num_threads, 0, cuda_stream>>>(
            H, L, Y_grad_data, X_data, padding_mask_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    const int64_t num_threads = (D < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    CombineRowwiseInternalGradientsKernel<T, T_ACC>
        <<<dim3(num_groups, B), num_threads, 0, cuda_stream>>>(
            H, num_groups, ds_data, db_data, gamma_data, dsw_data, dbw_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  {
    const int64_t num_threads = (L < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    GroupSequenceNormCUDABwdBHLKernel<T, T_ACC>
        <<<dim3(H, B), num_threads, 0, cuda_stream>>>(
            H, L, num_groups, Y_grad_data, X_data, count_data, mean_data,
            rstd_data, gamma_data, padding_mask_data, dsw_data, dbw_data,
            X_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
  GroupGammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, num_groups, mean_data, rstd_data, ds_data, db_data,
          gamma_grad_data, beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps, bool length_last) {
  const int64_t B = X.size(0);
  const int64_t H = X.size(length_last ? 1 : 2);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count =
      torch::empty({B}, X.options()
                            .dtype(torch::kInt64)
                            .memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(X.scalar_type(), true);
  torch::Tensor mean = torch::empty(
      {B, H},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor rstd = torch::empty(
      {B, H},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "SequenceNormCUDAFwd", [&]() {
        if (length_last) {
          SequenceNormCUDAFwdBHLImpl<scalar_t>(
              *(X.expect_contiguous()), *(gamma.expect_contiguous()),
              *(beta.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
              mean, rstd);
        } else {
          SequenceNormCUDAFwdBLHImpl<scalar_t>(
              *(X.expect_contiguous()), *(gamma.expect_contiguous()),
              *(beta.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
              mean, rstd);
        }
      });

  return std::make_tuple(Y, count, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "SequenceNormCUDABwd", [&]() {
        if (length_last) {
          SequenceNormCUDABwdBHLImpl<scalar_t>(
              *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
              *(count.expect_contiguous()), *(mean.expect_contiguous()),
              *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
              gamma_grad, beta_grad);
        } else {
          SequenceNormCUDABwdBLHImpl<scalar_t>(
              *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
              *(count.expect_contiguous()), *(mean.expect_contiguous()),
              *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
              gamma_grad, beta_grad);
        }
      });

  return std::make_tuple(X_grad, gamma_grad, beta_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GroupSequenceNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                         const torch::Tensor& beta,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, double eps, bool length_last) {
  const int64_t B = X.size(0);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count =
      torch::empty({B}, X.options()
                            .dtype(torch::kInt64)
                            .memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toOpMathType(X.scalar_type());
  torch::Tensor mean = torch::empty(
      {B, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor rstd = torch::empty(
      {B, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "GroupSequenceNormCUDAFwd",
      [&]() {
        if (length_last) {
          GroupSequenceNormCUDAFwdBHLImpl<scalar_t>(
              *(X.expect_contiguous()), *(gamma.expect_contiguous()),
              *(beta.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), num_groups, eps,
              Y, count, mean, rstd);
        } else {
          GroupSequenceNormCUDAFwdBLHImpl<scalar_t>(
              *(X.expect_contiguous()), *(gamma.expect_contiguous()),
              *(beta.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), num_groups, eps,
              Y, count, mean, rstd);
        }
      });

  return std::make_tuple(Y, count, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GroupSequenceNormCUDABwd(const torch::Tensor& Y_grad, const torch::Tensor& X,
                         const torch::Tensor& count, const torch::Tensor& mean,
                         const torch::Tensor& rstd, const torch::Tensor& gamma,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, bool length_last) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "GroupSequenceNormCUDABwd",
      [&]() {
        if (length_last) {
          GroupSequenceNormCUDABwdBHLImpl<scalar_t>(
              *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
              *(count.expect_contiguous()), *(mean.expect_contiguous()),
              *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), num_groups,
              X_grad, gamma_grad, beta_grad);
        } else {
          GroupSequenceNormCUDABwdBLHImpl<scalar_t>(
              *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
              *(count.expect_contiguous()), *(mean.expect_contiguous()),
              *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), num_groups,
              X_grad, gamma_grad, beta_grad);
        }
      });

  return std::make_tuple(X_grad, gamma_grad, beta_grad);
}

}  // namespace ops
}  // namespace megalodon
