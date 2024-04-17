#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>

#include <type_traits>

#include "blas.h"
#include "complex_utils.cuh"
#include "cuda_utils.cuh"
#include "ops/ema_hidden.h"
#include "reduce.cuh"
#include "utils.h"

namespace megalodon {
namespace ops {

namespace {

constexpr int64_t kMaxNumThreads = 256;

template <typename T, typename T_ACC>
__global__ void EMAHiddenBatchSize1CUDAFwdKernel(
    int64_t N, int64_t L, const T* __restrict__ x, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h,
    c10::complex<T_ACC>* __restrict__ y) {
  __shared__ T_ACC sum_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T_ACC>* sum_shared_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(sum_shared);

  const int64_t i = blockIdx.x;
  const int64_t k = blockIdx.y;
  const T_ACC p_v = p[i * N + k];
  const c10::complex<T_ACC> log_q_v = log_q[i * N + k];

  c10::complex<T_ACC> sum(T_ACC(0));
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const T_ACC x_v = static_cast<T_ACC>(x[i * L + j]);
    const c10::complex<T_ACC> qw =
        complex_utils::Exp(log_q_v * static_cast<T_ACC>(L - j - 1));
    sum += qw * x_v;
  }
  sum = reduce::BlockReduce(sum, sum_shared_ptr);

  if (threadIdx.x == 0) {
    sum *= p_v;
    if (h != nullptr) {
      const c10::complex<T_ACC> qw =
          complex_utils::Exp(log_q_v * static_cast<T_ACC>(L));
      sum += qw * h[i * N + k];
    }
    y[i * N + k] = sum;
  }
}

template <typename T, typename T_ACC, int64_t N>
__global__ void EMAHiddenBatchSize1CUDAFwdKernel(
    int64_t L, const T* __restrict__ x, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h,
    c10::complex<T_ACC>* __restrict__ y) {
  extern __shared__ float shm[];
  T_ACC* p_ptr = reinterpret_cast<T_ACC*>(shm);
  c10::complex<T_ACC>* q_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(p_ptr + N);
  c10::complex<T_ACC>* h_ptr = q_ptr + N;
  c10::complex<T_ACC>* sum_shared = h_ptr + N;
  c10::complex<T_ACC>* q_pow = sum_shared + cuda_utils::kWarpSize;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    p_ptr[threadIdx.x] = p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    h_ptr[threadIdx.x] =
        h == nullptr ? c10::complex<T_ACC>(0) : h[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  c10::complex<T_ACC> sum[N];
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    sum[k] = c10::complex<T_ACC>(0);
  }
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const int64_t idx = L - j - 1;
    const int64_t row = idx / chunk_size;
    const int64_t col = idx % chunk_size;
    const T_ACC x_acc = static_cast<T_ACC>(x[i * L + j]);
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          q_pow[k * chunk_size + col] * c_ptr[k * num_chunks + row];
      sum[k] += qw * x_acc;
    }
  }
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    sum[k] = reduce::BlockReduce(sum[k], sum_shared);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      sum[k] *= p_ptr[k];
      if (h != nullptr) {
        const c10::complex<T_ACC> qw =
            complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(L));
        sum[k] += qw * h_ptr[k];
      }
      y[i * N + k] = sum[k];
    }
  }
}

template <typename T, typename T_ACC, int64_t B>
__global__ void EMAHiddenCUDAFwdKernel(
    int64_t D, int64_t N, int64_t L, const T* __restrict__ x,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h,
    c10::complex<T_ACC>* __restrict__ y) {
  __shared__ T_ACC sum_shared[B][cuda_utils::kWarpSize * 2];

  const int64_t i = blockIdx.x;
  const int64_t k = blockIdx.y;

  const T_ACC p_v = p[i * N + k];
  const c10::complex<T_ACC> log_q_v = log_q[i * N + k];

  c10::complex<T_ACC> sum[B];
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
    sum[b] = c10::complex<T_ACC>(0);
  }

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const c10::complex<T_ACC> qw =
        complex_utils::Exp(log_q_v * static_cast<T_ACC>(L - j - 1));
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      const T_ACC x_v = static_cast<T_ACC>(x[(b * D + i) * L + j]);
      sum[b] += qw * x_v;
    }
  }
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
    sum[b] = reduce::BlockReduce(
        sum[b], reinterpret_cast<c10::complex<T_ACC>*>(sum_shared[b]));
  }

  if (threadIdx.x == 0) {
    const c10::complex<T_ACC> qw =
        h == nullptr ? c10::complex<T_ACC>(T_ACC(0))
                     : complex_utils::Exp(log_q_v * static_cast<T_ACC>(L));
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      sum[b] = sum[b] * p_v + (h == nullptr ? c10::complex<T_ACC>(T_ACC(0))
                                            : qw * h[(b * D + i) * N + k]);
      y[(b * D + i) * N + k] = sum[b];
    }
  }
}

template <typename T, typename T_ACC, int64_t B, int64_t N>
__global__ void EMAHiddenCUDAFwdKernel(
    int64_t D, int64_t L, const T* __restrict__ x, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h,
    c10::complex<T_ACC>* __restrict__ y) {
  extern __shared__ float shm[];
  T_ACC* p_ptr = reinterpret_cast<T_ACC*>(shm);
  c10::complex<T_ACC>* q_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(p_ptr + N);
  c10::complex<T_ACC>* h_ptr = q_ptr + N;
  c10::complex<T_ACC>* sum_shared = h_ptr + B * N;
  c10::complex<T_ACC>* q_pow = sum_shared + cuda_utils::kWarpSize;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    p_ptr[threadIdx.x] = p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t b = j / N;
    const int64_t k = j % N;
    h_ptr[j] = h == nullptr ? c10::complex<T_ACC>(0) : h[(b * D + i) * N + k];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  c10::complex<T_ACC> sum[N];
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      sum[k] = c10::complex<T_ACC>(0);
    }
    for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
      const int64_t idx = L - j - 1;
      const int64_t row = idx / chunk_size;
      const int64_t col = idx % chunk_size;
      const T_ACC x_acc = static_cast<T_ACC>(x[(b * D + i) * L + j]);
#pragma unroll
      for (int64_t k = 0; k < N; ++k) {
        const c10::complex<T_ACC> qw =
            q_pow[k * chunk_size + col] * c_ptr[k * num_chunks + row];
        sum[k] += qw * x_acc;
      }
    }
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      sum[k] = reduce::BlockReduce(sum[k], sum_shared);
    }

    if (threadIdx.x == 0) {
#pragma unroll
      for (int64_t k = 0; k < N; ++k) {
        sum[k] *= p_ptr[k];
        if (h != nullptr) {
          const c10::complex<T_ACC> qw =
              complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(L));
          sum[k] += qw * h_ptr[b * N + k];
        }
        y[(b * D + i) * N + k] = sum[k];
      }
    }
  }
}

template <typename T>
__global__ void EMAHiddenWeightCUDAFwdKernel(
    int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    c10::complex<T>* __restrict__ v) {
  const int64_t i = blockIdx.x;
  const T p_v = p[i];
  const c10::complex<T> log_q_v = log_q[i];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const c10::complex<T> qw =
        complex_utils::Exp(log_q_v * static_cast<T>(L - j - 1));
    v[i * L + j] = p_v * qw;
  }
}

template <typename T>
__global__ void EMAHiddenBiasCUDAFwdKernel(
    int64_t B, int64_t D, int64_t N, int64_t L,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ h, c10::complex<T>* __restrict__ y) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D * N) {
    return;
  }
  const c10::complex<T> qw = complex_utils::Exp(log_q[i] * static_cast<T>(L));
  for (int64_t b = 0; b < B; ++b) {
    y[b * D * N + i] += h[b * D * N + i] * qw;
  }
}

template <typename T, typename T_ACC>
__global__ void EMAHiddenInputBatchSize1CUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    T* __restrict__ x_grad) {
  extern __shared__ float shm[];
  c10::complex<T_ACC>* w_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(shm);  // y_grad.conj() * p
  c10::complex<T_ACC>* q_ptr = w_ptr + N;
  c10::complex<T_ACC>* q_pow = q_ptr + N;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    w_ptr[threadIdx.x] =
        std::conj(y_grad[i * N + threadIdx.x]) * p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const int64_t idx = L - j - 1;
    const int64_t row = idx / chunk_size;
    const int64_t col = idx % chunk_size;
    T_ACC sum = T_ACC(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          q_pow[k * chunk_size + col] * c_ptr[k * num_chunks + row];
      sum += complex_utils::RealOfProduct(w_ptr[k], qw);
    }
    x_grad[i * L + j] = static_cast<T>(sum);
  }
}

template <typename T, typename T_ACC, int64_t N>
__global__ void EMAHiddenInputBatchSize1CUDABwdKernel(
    int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    T* __restrict__ x_grad) {
  extern __shared__ float shm[];
  c10::complex<T_ACC>* w_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(shm);  // y_grad.conj() * p
  c10::complex<T_ACC>* q_ptr = w_ptr + N;
  c10::complex<T_ACC>* q_pow = q_ptr + N;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    w_ptr[threadIdx.x] =
        std::conj(y_grad[i * N + threadIdx.x]) * p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const int64_t idx = L - j - 1;
    const int64_t row = idx / chunk_size;
    const int64_t col = idx % chunk_size;
    T_ACC sum = T_ACC(0);
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          q_pow[k * chunk_size + col] * c_ptr[k * num_chunks + row];
      sum += complex_utils::RealOfProduct(w_ptr[k], qw);
    }
    x_grad[i * L + j] = static_cast<T>(sum);
  }
}

template <typename T, typename T_ACC>
__global__ void EMAHiddenBatchSize1CUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T* __restrict__ x, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h, T_ACC* __restrict__ p_grad,
    c10::complex<T_ACC>* __restrict__ q_grad, c10::complex<T_ACC>* h_grad) {
  __shared__ T_ACC sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T_ACC sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T_ACC>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(sum1_shared);
  c10::complex<T_ACC>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const int64_t k = blockIdx.y;
  const c10::complex<T_ACC> dy = std::conj(y_grad[i * N + k]);
  const T_ACC p_v = p[i * N + k];
  const c10::complex<T_ACC> log_q_v = log_q[i * N + k];
  const c10::complex<T_ACC> q_v = complex_utils::Exp(log_q_v);

  c10::complex<T_ACC> sum1(T_ACC(0));
  c10::complex<T_ACC> sum2(T_ACC(0));
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const T_ACC x_v = static_cast<T_ACC>(x[i * L + j]);
    const c10::complex<T_ACC> qw1 =
        j == L - 1
            ? c10::complex<T_ACC>(T_ACC(0))
            : complex_utils::Exp(log_q_v * static_cast<T_ACC>(L - j - 2));
    const c10::complex<T_ACC> qw2 =
        j == L - 1 ? c10::complex<T_ACC>(T_ACC(1)) : qw1 * q_v;
    sum1 += x_v * qw2;
    sum2 += x_v * qw1 * static_cast<T_ACC>(L - j - 1);
  }
  sum1 = reduce::BlockReduce(sum1, sum1_shared_ptr);
  sum2 = reduce::BlockReduce(sum2, sum2_shared_ptr);

  if (threadIdx.x == 0) {
    p_grad[i * N + k] = complex_utils::RealOfProduct(dy, sum1);
    q_grad[i * N + k] = std::conj(dy * p_v * sum2);
    if (h != nullptr) {
      const c10::complex<T_ACC> qw1 =
          complex_utils::Exp(log_q_v * static_cast<T_ACC>(L - 1));
      const c10::complex<T_ACC> qw2 = qw1 * q_v;
      q_grad[i * N + k] +=
          std::conj(dy * h[i * N + k] * qw1 * static_cast<T_ACC>(L));
      h_grad[i * N + k] = std::conj(dy * qw2);
    }
  }
}

template <typename T, typename T_ACC, int64_t N>
__global__ void EMAHiddenBatchSize1CUDABwdKernel(
    int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T* __restrict__ x, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h, T_ACC* __restrict__ p_grad,
    c10::complex<T_ACC>* __restrict__ q_grad, c10::complex<T_ACC>* h_grad) {
  extern __shared__ float shm[];
  T_ACC* p_ptr = reinterpret_cast<T_ACC*>(shm);
  c10::complex<T_ACC>* d_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(p_ptr + N);
  c10::complex<T_ACC>* q_ptr = d_ptr + N;
  c10::complex<T_ACC>* h_ptr = q_ptr + N;
  c10::complex<T_ACC>* sum1_shared = h_ptr + N;
  c10::complex<T_ACC>* sum2_shared = sum1_shared + cuda_utils::kWarpSize;
  c10::complex<T_ACC>* q_pow = sum2_shared + cuda_utils::kWarpSize;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    p_ptr[threadIdx.x] = p[i * N + threadIdx.x];
    d_ptr[threadIdx.x] = std::conj(y_grad[i * N + threadIdx.x]);
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    h_ptr[threadIdx.x] =
        h == nullptr ? c10::complex<T_ACC>(0) : h[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  c10::complex<T_ACC> sum1[N];
  c10::complex<T_ACC> sum2[N];
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    sum1[k] = c10::complex<T_ACC>(0);
    sum2[k] = c10::complex<T_ACC>(0);
  }
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const int64_t idx = L - j - 1;
    const int64_t r2 = idx / chunk_size;
    const int64_t c2 = idx % chunk_size;
    const int64_t r1 = c2 > 0 ? r2 : r2 - 1;
    const int64_t c1 = c2 > 0 ? c2 - 1 : chunk_size - 1;
    const T_ACC x_acc = static_cast<T_ACC>(x[i * L + j]);
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw1 =
          j == L - 1 ? c10::complex<T_ACC>(T_ACC(0))
                     : q_pow[k * chunk_size + c1] * c_ptr[k * num_chunks + r1];
      const c10::complex<T_ACC> qw2 =
          q_pow[k * chunk_size + c2] * c_ptr[k * num_chunks + r2];
      sum1[k] += x_acc * qw2;
      sum2[k] += x_acc * qw1 * static_cast<T_ACC>(idx);
    }
  }
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    sum1[k] = reduce::BlockReduce(sum1[k], sum1_shared);
    sum2[k] = reduce::BlockReduce(sum2[k], sum2_shared);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      p_grad[i * N + k] = complex_utils::RealOfProduct(d_ptr[k], sum1[k]);
      if (h == nullptr) {
        q_grad[i * N + k] = std::conj(d_ptr[k] * p_ptr[k] * sum2[k]);
      } else {
        const c10::complex<T_ACC> qw1 =
            complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(L - 1));
        const c10::complex<T_ACC> qw2 = qw1 * q_pow[k * chunk_size + 1];
        q_grad[i * N + k] =
            std::conj(d_ptr[k] * (p_ptr[k] * sum2[k] +
                                  h_ptr[k] * qw1 * static_cast<T_ACC>(L)));
        h_grad[i * N + k] = std::conj(d_ptr[k] * qw2);
      }
    }
  }
}

template <typename T, typename T_ACC, int64_t B>
__global__ void EMAHiddenInputCUDABwdKernel(
    int64_t D, int64_t N, int64_t L,
    const c10::complex<T_ACC>* __restrict__ y_grad, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q, T* __restrict__ x_grad) {
  extern __shared__ float shm[];
  c10::complex<T_ACC>* w_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(shm);  // y_grad.conj() * p
  c10::complex<T_ACC>* q_ptr = w_ptr + B * N;
  c10::complex<T_ACC>* q_pow = q_ptr + N;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t b = j / N;
    const int64_t k = j % N;
    w_ptr[j] = std::conj(y_grad[(b * D + i) * N + k]) * p[i * N + k];
  }
  __syncthreads();

  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  T_ACC sum[B];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const int64_t idx = L - j - 1;
    const int64_t row = idx / chunk_size;
    const int64_t col = idx % chunk_size;

#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      sum[b] = T_ACC(0);
    }
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          q_pow[k * chunk_size + col] * c_ptr[k * num_chunks + row];
#pragma unroll
      for (int64_t b = 0; b < B; ++b) {
        sum[b] += complex_utils::RealOfProduct(w_ptr[b * N + k], qw);
      }
    }
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      x_grad[(b * D + i) * L + j] = static_cast<T>(sum[b]);
    }
  }
}

template <typename T, typename T_ACC, int64_t B, int64_t N>
__global__ void EMAHiddenInputCUDABwdKernel(
    int64_t D, int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    T* __restrict__ x_grad) {
  extern __shared__ float shm[];
  c10::complex<T_ACC>* w_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(shm);  // y_grad.conj() * p
  c10::complex<T_ACC>* q_ptr = w_ptr + B * N;
  c10::complex<T_ACC>* q_pow = q_ptr + N;
  c10::complex<T_ACC>* c_ptr = q_pow + N * blockDim.x;

  const int64_t chunk_size = blockDim.x;
  const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t b = j / N;
    const int64_t k = j % N;
    w_ptr[j] = std::conj(y_grad[(b * D + i) * N + k]) * p[i * N + k];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * chunk_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(threadIdx.x));
  }
  for (int64_t j = threadIdx.x; j < num_chunks; j += blockDim.x) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      c_ptr[k * num_chunks + j] =
          complex_utils::Exp(q_ptr[k] * static_cast<T_ACC>(j * chunk_size));
    }
  }
  __syncthreads();

  T_ACC sum[B];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const int64_t idx = L - j - 1;
    const int64_t row = idx / chunk_size;
    const int64_t col = idx % chunk_size;

#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      sum[b] = T_ACC(0);
    }
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          q_pow[k * chunk_size + col] * c_ptr[k * num_chunks + row];
#pragma unroll
      for (int64_t b = 0; b < B; ++b) {
        sum[b] += complex_utils::RealOfProduct(w_ptr[b * N + k], qw);
      }
    }
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      x_grad[(b * D + i) * L + j] = static_cast<T>(sum[b]);
    }
  }
}

template <typename T, typename T_ACC, int64_t B>
__global__ void EMAHiddenCUDABwdKernel(
    int64_t D, int64_t N, int64_t L,
    const c10::complex<T_ACC>* __restrict__ y_grad, const T* __restrict__ x,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h, T_ACC* __restrict__ p_grad,
    c10::complex<T_ACC>* __restrict__ q_grad, c10::complex<T_ACC>* h_grad) {
  __shared__ T_ACC sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T_ACC sum2_shared[cuda_utils::kWarpSize * 2];

  const int64_t i = blockIdx.x;
  const int64_t k = blockIdx.y;
  const T_ACC p_v = p[i * N + k];
  const c10::complex<T_ACC> log_q_v = log_q[i * N + k];
  const c10::complex<T_ACC> q_v = complex_utils::Exp(log_q_v);

  c10::complex<T_ACC> sum1[B];
  c10::complex<T_ACC> sum2[B];
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
    sum1[b] = c10::complex<T_ACC>(0);
    sum2[b] = c10::complex<T_ACC>(0);
  }

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const c10::complex<T_ACC> qw1 =
        j == L - 1
            ? c10::complex<T_ACC>(T_ACC(0))
            : complex_utils::Exp(log_q_v * static_cast<T_ACC>(L - j - 2));
    const c10::complex<T_ACC> qw2 =
        j == L - 1 ? c10::complex<T_ACC>(T_ACC(1)) : qw1 * q_v;
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      const T_ACC x_v = static_cast<T_ACC>(x[(b * D + i) * L + j]);
      sum1[b] += x_v * qw2;
      sum2[b] += x_v * qw1 * static_cast<T_ACC>(L - j - 1);
    }
  }
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
    sum1[b] = reduce::BlockReduce(
        sum1[b], reinterpret_cast<c10::complex<T_ACC>*>(sum1_shared));
    sum2[b] = reduce::BlockReduce(
        sum2[b], reinterpret_cast<c10::complex<T_ACC>*>(sum2_shared));
  }

  if (threadIdx.x == 0) {
    const c10::complex<T_ACC> qw1 =
        h == nullptr ? c10::complex<T_ACC>(T_ACC(0))
                     : complex_utils::Exp(log_q_v * static_cast<T_ACC>(L - 1));
    const c10::complex<T_ACC> qw2 = qw1 * q_v;
    T_ACC dp = T_ACC(0);
    c10::complex<T_ACC> dq(T_ACC(0));
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      const c10::complex<T_ACC> dy = std::conj(y_grad[(b * D + i) * N + k]);
      dp += complex_utils::RealOfProduct(dy, sum1[b]);
      dq += dy * p_v * sum2[b];
      if (h != nullptr) {
        dq += dy * h[i * N + k] * qw1 * static_cast<T_ACC>(L);
        h_grad[(b * D + i) * N + k] = std::conj(dy * qw2);
      }
    }
    p_grad[i * N + k] = dp;
    q_grad[i * N + k] = std::conj(dq);
  }
}

template <typename T>
__global__ void EMAHiddenWeightCUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T>* __restrict__ v_grad,
    const T* __restrict__ p, const c10::complex<T>* __restrict__ log_q,
    T* __restrict__ p_grad, c10::complex<T>* __restrict__ q_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  T* sum1_shared_ptr = sum1_shared;
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const T p_v = p[i];
  const c10::complex<T> log_q_v = log_q[i];
  const c10::complex<T> q_v = complex_utils::Exp(log_q_v);

  T sum1 = T(0);
  c10::complex<T> sum2(T(0));
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const c10::complex<T> dv = v_grad[i * L + j];
    const c10::complex<T> qw1 =
        j == L - 1 ? c10::complex<T>(T(0))
                   : complex_utils::Exp(log_q_v * static_cast<T>(L - j - 2));
    const c10::complex<T> qw2 = j == L - 1 ? c10::complex<T>(T(1)) : qw1 * q_v;

    sum1 += complex_utils::RealOfProduct(qw2, dv);
    sum2 += qw1 * (dv * static_cast<T>(L - j - 1));
  }
  sum1 = reduce::BlockReduce(sum1, sum1_shared_ptr);
  sum2 = reduce::BlockReduce(sum2, sum2_shared_ptr);

  if (threadIdx.x == 0) {
    p_grad[i] = sum1;
    q_grad[i] = std::conj(p_v * sum2);
  }
}

template <typename T>
__global__ void EMAHiddenBiasCUDABwdKernel(
    int64_t B, int64_t D, int64_t N, int64_t L,
    const c10::complex<T>* __restrict__ y_grad,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ h, c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ h_grad) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D * N) {
    return;
  }

  const c10::complex<T> log_q_v = log_q[i];
  const c10::complex<T> q_v = complex_utils::Exp(log_q_v);
  const c10::complex<T> qw1 =
      complex_utils::Exp(log_q_v * static_cast<T>(L - 1));
  const c10::complex<T> qw2 = qw1 * q_v;
  c10::complex<T> sum(T(0));
  for (int64_t b = 0; b < B; ++b) {
    const int64_t index = b * D * N + i;
    const c10::complex<T> dy = std::conj(y_grad[index]);
    sum += dy * h[index];
    h_grad[index] = std::conj(dy * qw2);
  }
  q_grad[i] += std::conj(sum * qw1 * static_cast<T>(L));
}

#define DISPATCH_EMA_HIDDEN_FWD_KERNEL(KernelFunc, T, T_ACC, num_threads,   \
                                       shm_size, cuda_stream, D, N, L, ...) \
  do {                                                                      \
    if (N == 16) {                                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 16>, D, num_threads,    \
                               shm_size, cuda_stream, L, __VA_ARGS__);      \
    } else {                                                                \
      KernelFunc<T, T_ACC>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    }                                                                       \
  } while (false)

#define DISPATCH_EMA_HIDDEN_BATCH_FWD_KERNEL(                                  \
    KernelFunc, T, T_ACC, num_threads, shm_size, cuda_stream, B, D, N, L, ...) \
  do {                                                                         \
    if (B == 2) {                                                              \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 2><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else if (B == 3) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 3, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 3><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else if (B == 4) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 4><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else if (B == 5) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 5, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 5><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else if (B == 6) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 6, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 6><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else if (B == 7) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 7, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 7><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else if (B == 8) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        KernelFunc<T, T_ACC, 8><<<dim3(D, N), num_threads, 0, cuda_stream>>>(  \
            D, N, L, __VA_ARGS__);                                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false);                                                      \
    }                                                                          \
  } while (false)

#define DISPATCH_EMA_HIDDEN_INPUT_BWD_KERNEL(                               \
    KernelFunc, T, T_ACC, num_threads, shm_size, cuda_stream, D, N, L, ...) \
  do {                                                                      \
    if (N == 16) {                                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 16>, D, num_threads,    \
                               shm_size, cuda_stream, L, __VA_ARGS__);      \
    } else {                                                                \
      KernelFunc<T, T_ACC>                                                  \
          <<<D, num_threads, shm_size, cuda_stream>>>(N, L, __VA_ARGS__);   \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    }                                                                       \
  } while (false)

#define DISPATCH_EMA_HIDDEN_INPUT_BATCH_BWD_KERNEL(                            \
    KernelFunc, T, T_ACC, num_threads, shm_size, cuda_stream, B, D, N, L, ...) \
  do {                                                                         \
    if (B == 2) {                                                              \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else if (B == 3) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 3, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 3>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else if (B == 4) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else if (B == 5) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 5, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 5>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else if (B == 6) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 6, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 6>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else if (B == 7) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 7, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 7>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else if (B == 8) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16>, D, num_threads,  \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8>, D, num_threads,      \
                                 shm_size, cuda_stream, D, N, L, __VA_ARGS__); \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false);                                                      \
    }                                                                          \
  } while (false)

#define DISPATCH_EMA_HIDDEN_BWD_KERNEL(KernelFunc, T, T_ACC, num_threads,   \
                                       shm_size, cuda_stream, D, N, L, ...) \
  do {                                                                      \
    if (N == 16) {                                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 16>, D, num_threads,    \
                               shm_size, cuda_stream, L, __VA_ARGS__);      \
    } else {                                                                \
      KernelFunc<T, T_ACC>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    }                                                                       \
  } while (false)

#define DISPATCH_EMA_HIDDEN_BATCH_BWD_KERNEL(                                  \
    KernelFunc, T, T_ACC, num_threads, shm_size, cuda_stream, B, D, N, L, ...) \
  do {                                                                         \
    if (B == 2) {                                                              \
      KernelFunc<T, T_ACC, 2>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else if (B == 3) {                                                       \
      KernelFunc<T, T_ACC, 3>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else if (B == 4) {                                                       \
      KernelFunc<T, T_ACC, 4>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else if (B == 5) {                                                       \
      KernelFunc<T, T_ACC, 5>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else if (B == 6) {                                                       \
      KernelFunc<T, T_ACC, 6>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else if (B == 7) {                                                       \
      KernelFunc<T, T_ACC, 7>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else if (B == 8) {                                                       \
      KernelFunc<T, T_ACC, 8>                                                  \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    } else {                                                                   \
      TORCH_CHECK(false);                                                      \
    }                                                                          \
  } while (false)

template <typename T>
void EMAHiddenCUDAFwdImpl(const torch::Tensor& x, const torch::Tensor& p,
                          const torch::Tensor& log_q, const torch::Tensor& h,
                          torch::Tensor& y, c10::optional<torch::Tensor>& v) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  const int64_t L = x.size(2);

  const T* x_data = x.data_ptr<T>();
  const T_ACC* p_data = p.data_ptr<T_ACC>();
  const c10::complex<T_ACC>* log_q_data = log_q.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* h_data =
      h.defined() ? h.data_ptr<c10::complex<T_ACC>>() : nullptr;
  c10::complex<T_ACC>* y_data = y.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);

  if (B == 1) {
    const int64_t num_chunks = utils::DivUp(L, num_threads);
    const int64_t shm_size =
        N * sizeof(T_ACC) +
        (2 * N + cuda_utils::kWarpSize + N * num_threads + N * num_chunks) *
            sizeof(c10::complex<T_ACC>);
    DISPATCH_EMA_HIDDEN_FWD_KERNEL(EMAHiddenBatchSize1CUDAFwdKernel, T, T_ACC,
                                   num_threads, shm_size, cuda_stream, D, N, L,
                                   x_data, p_data, log_q_data, h_data, y_data);
    return;
  }
  if (B <= 8) {
    const int64_t num_chunks = utils::DivUp(L, num_threads);
    const int64_t shm_size =
        N * sizeof(T_ACC) +
        (N + B * N + cuda_utils::kWarpSize + N * num_threads + N * num_chunks) *
            sizeof(c10::complex<T_ACC>);
    DISPATCH_EMA_HIDDEN_BATCH_FWD_KERNEL(
        EMAHiddenCUDAFwdKernel, T, T_ACC, num_threads, shm_size, cuda_stream, B,
        D, N, L, x_data, p_data, log_q_data, h_data, y_data);
    return;
  }

  torch::Tensor x_c = x.to(log_q.scalar_type());
  v = c10::make_optional(torch::empty({D, N, L}, log_q.options()));
  const c10::complex<T_ACC>* x_c_data = x_c.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* v_data = v->data_ptr<c10::complex<T_ACC>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T_ACC> kAlpha(1);
  const c10::complex<T_ACC> kBeta(0);

  EMAHiddenWeightCUDAFwdKernel<T_ACC>
      <<<D * N, num_threads, 0, cuda_stream>>>(L, p_data, log_q_data, v_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  blas::GemmStridedBatchedCUDA<c10::complex<T_ACC>>(
      handle, blas::TransposeOp::kN, blas::TransposeOp::kT, D, B, N, L, kAlpha,
      x_c_data, D * L, L, v_data, L, N * L, kBeta, y_data, D * N, N);

  if (h_data != nullptr) {
    const int64_t M = utils::DivUp(D * N, cuda_utils::kCUDANumThreads);
    EMAHiddenBiasCUDAFwdKernel<T_ACC>
        <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, D, N, L, log_q_data, h_data, y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T>
void EMAHiddenCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& x,
                          const torch::Tensor& p, const torch::Tensor& log_q,
                          const torch::Tensor& h, const torch::Tensor& v,
                          torch::Tensor& x_grad, torch::Tensor& p_grad,
                          torch::Tensor& q_grad,
                          c10::optional<torch::Tensor>& h_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  const int64_t L = x.size(2);
  TORCH_CHECK(N <= cuda_utils::kWarpSize);

  const c10::complex<T_ACC>* y_grad_data =
      y_grad.data_ptr<c10::complex<T_ACC>>();
  const T* x_data = x.data_ptr<T>();
  const T_ACC* p_data = p.data_ptr<T_ACC>();
  const c10::complex<T_ACC>* log_q_data = log_q.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* h_data =
      h.defined() ? h.data_ptr<c10::complex<T_ACC>>() : nullptr;
  T* x_grad_data = x_grad.data_ptr<T>();
  T_ACC* p_grad_data = p_grad.data_ptr<T_ACC>();
  c10::complex<T_ACC>* q_grad_data = q_grad.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* h_grad_data = nullptr;

  if (h.defined()) {
    h_grad = c10::make_optional(torch::empty_like(h));
    h_grad_data = h_grad->data_ptr<c10::complex<T_ACC>>();
  }

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);
  const int64_t num_chunks = utils::DivUp(L, num_threads);

  if (B == 1) {
    const int64_t dx_shm_size = (2 * N + N * num_threads + N * num_chunks) *
                                sizeof(c10::complex<T_ACC>);
    DISPATCH_EMA_HIDDEN_INPUT_BWD_KERNEL(EMAHiddenInputBatchSize1CUDABwdKernel,
                                         T, T_ACC, num_threads, dx_shm_size,
                                         cuda_stream, D, N, L, y_grad_data,
                                         p_data, log_q_data, x_grad_data);
    const int64_t dw_shm_size =
        N * sizeof(T_ACC) +
        (3 * N + 2 * cuda_utils::kWarpSize + N * num_threads + N * num_chunks) *
            sizeof(c10::complex<T_ACC>);
    DISPATCH_EMA_HIDDEN_BWD_KERNEL(
        EMAHiddenBatchSize1CUDABwdKernel, T, T_ACC, num_threads, dw_shm_size,
        cuda_stream, D, N, L, y_grad_data, x_data, p_data, log_q_data, h_data,
        p_grad_data, q_grad_data, h_grad_data);
    return;
  }

  if (B <= 8) {
    const int64_t num_chunks = utils::DivUp(L, num_threads);
    const int64_t shm_size = (B * N + N + N * num_threads + N * num_chunks) *
                             sizeof(c10::complex<T_ACC>);
    DISPATCH_EMA_HIDDEN_INPUT_BATCH_BWD_KERNEL(
        EMAHiddenInputCUDABwdKernel, T, T_ACC, num_threads, shm_size,
        cuda_stream, B, D, N, L, y_grad_data, p_data, log_q_data, x_grad_data);
    // TODO: Optimize this.
    DISPATCH_EMA_HIDDEN_BATCH_BWD_KERNEL(
        EMAHiddenCUDABwdKernel, T, T_ACC, num_threads, 0, cuda_stream, B, D, N,
        L, y_grad_data, x_data, p_data, log_q_data, h_data, p_grad_data,
        q_grad_data, h_grad_data);
    return;
  }

  TORCH_CHECK(v.defined());
  torch::Tensor y_grad_conj = torch::conj_physical(y_grad);
  torch::Tensor x_c = x.to(v.scalar_type());
  torch::Tensor x_grad_c = torch::empty({B, D, L}, v.options());
  torch::Tensor v_grad = torch::empty_like(v);
  const c10::complex<T_ACC>* y_grad_conj_data =
      y_grad_conj.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* x_c_data = x_c.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* v_data = v.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* x_grad_c_data = x_grad_c.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* v_grad_data = v_grad.data_ptr<c10::complex<T_ACC>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T_ACC> kAlpha(1);
  const c10::complex<T_ACC> kBeta(0);

  blas::GemmStridedBatchedCUDA<c10::complex<T_ACC>>(
      handle, blas::TransposeOp::kN, blas::TransposeOp::kN, D, B, L, N, kAlpha,
      y_grad_conj_data, D * N, N, v_data, L, N * L, kBeta, x_grad_c_data, D * L,
      L);
  blas::GemmStridedBatchedCUDA<c10::complex<T_ACC>>(
      handle, blas::TransposeOp::kC, blas::TransposeOp::kN, D, N, L, B, kAlpha,
      y_grad_data, D * N, N, x_c_data, D * L, L, kBeta, v_grad_data, L, N * L);

  // TODO: Optimize this.
  x_grad = torch::real(x_grad_c).to(x.scalar_type());

  EMAHiddenWeightCUDABwdKernel<T_ACC><<<D * N, num_threads, 0, cuda_stream>>>(
      N, L, v_grad_data, p_data, log_q_data, p_grad_data, q_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (h_data != nullptr) {
    const int64_t M = utils::DivUp(D * N, cuda_utils::kCUDANumThreads);
    EMAHiddenBiasCUDABwdKernel<T_ACC>
        <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, D, N, L, y_grad_data, log_q_data, h_data, q_grad_data,
            h_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

#undef DISPATCH_EMA_HIDDEN_FWD_KERNEL
#undef DISPATCH_EMA_HIDDEN_BATCH_FWD_KERNEL
#undef DISPATCH_EMA_HIDDEN_INPUT_BWD_KERNEL
#undef DISPATCH_EMA_HIDDEN_INPUT_BATCH_BWD_KERNEL
#undef DISPATCH_EMA_HIDDEN_BWD_KERNEL
#undef DISPATCH_EMA_HIDDEN_BATCH_BWD_KERNEL

}  // namespace

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenCUDAFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h) {
  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor y = torch::empty(
      {B, D, N}, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> v = c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "EMAHiddenCUDAFwd", [&]() {
        EMAHiddenCUDAFwdImpl<scalar_t>(
            *(x.expect_contiguous()), *(p.expect_contiguous()),
            *(log_q.expect_contiguous()), *(h_maybe_owned->expect_contiguous()),
            y, v);
      });

  return std::make_tuple<torch::Tensor, c10::optional<torch::Tensor>>(
      std::move(y), std::move(v));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                 const torch::Tensor& p, const torch::Tensor& log_q,
                 const c10::optional<torch::Tensor>& h,
                 const c10::optional<torch::Tensor>& v) {
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  c10::MaybeOwned<torch::Tensor> v_maybe_owned =
      at::borrow_from_optional_tensor(v);
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
      at::kHalf, at::kBFloat16, x.scalar_type(), "EMAHiddenCUDABwd", [&]() {
        EMAHiddenCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(p.expect_contiguous()), *(log_q.expect_contiguous()),
            *(h_maybe_owned->expect_contiguous()),
            *(v_maybe_owned->expect_contiguous()), x_grad, p_grad, q_grad,
            h_grad);
      });

  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>>(
      std::move(x_grad), std::move(p_grad), std::move(q_grad),
      std::move(h_grad));
}

}  // namespace ops
}  // namespace megalodon
