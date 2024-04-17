#include <ATen/DeviceGuard.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <cmath>
#include <type_traits>

#include "blas.h"
#include "complex_utils.cuh"
#include "cuda_utils.cuh"
#include "ops/ema_parameters.h"
#include "reduce.cuh"
#include "register_utils.cuh"

namespace megalodon {
namespace ops {

namespace {

constexpr int64_t kMaxNumThreads = 256;

template <typename T>
__global__ void EMAVandermondeCUDAFwdKernel(
    int64_t N, int64_t L, const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    c10::complex<T>* __restrict__ v) {
  const int64_t i = blockIdx.x;
  const int64_t k = blockIdx.y;
  const c10::complex<T> log_q_v = log_q[i * N + k];
  const c10::complex<T> gamma_v = gamma[i * N + k];

  for (int64_t j = threadIdx.x; j <= L; j += blockDim.x) {
    const c10::complex<T> qw = complex_utils::Exp(log_q_v * static_cast<T>(j));
    v[(i * N + k) * (L + 1) + j] = qw * gamma_v;
  }
}

template <typename T>
__global__ void EMAWeightCUDAFwdKernel(
    int64_t N, int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ weight) {
  __shared__ T q_shared[cuda_utils::kWarpSize * 2];
  __shared__ T u_shared[cuda_utils::kWarpSize * 2];  // p * gamma
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(q_shared);
  c10::complex<T>* u_ptr = reinterpret_cast<c10::complex<T>*>(u_shared);

  const int64_t i = blockIdx.x;
  T* w_ptr = weight + i * L;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    u_ptr[threadIdx.x] = p[i * N + threadIdx.x] * gamma[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum = T(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw =
          complex_utils::Exp(q_ptr[k] * static_cast<T>(j));
      sum += complex_utils::RealOfProduct(qw, u_ptr[k]);
    }
    w_ptr[j] = sum;
  }
}

template <typename T, int64_t N>
__global__ void EMAWeightCUDAFwdKernel(
    int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ weight) {
  extern __shared__ float shm[];
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(shm);
  c10::complex<T>* u_ptr = q_ptr + N;  // p * gamma
  c10::complex<T>* q_pow = u_ptr + N;

  const int64_t inner_size = blockDim.x + 1;
  const int64_t i = blockIdx.x;
  T* w_ptr = weight + i * L;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    u_ptr[threadIdx.x] = p[i * N + threadIdx.x] * gamma[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * inner_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T>(threadIdx.x));
  }
  if (threadIdx.x < N) {
    q_pow[threadIdx.x * inner_size + blockDim.x] =
        complex_utils::Exp(q_ptr[threadIdx.x] * static_cast<T>(blockDim.x));
  }
  __syncthreads();

  c10::complex<T> c[N];
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    c[k] = c10::complex<T>(1);
  }

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum = T(0);
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw = q_pow[k * inner_size + threadIdx.x] * c[k];
      sum += complex_utils::RealOfProduct(qw, u_ptr[k]);
      c[k] *= q_pow[k * inner_size + blockDim.x];
    }
    w_ptr[j] = sum;
  }
}

template <typename T>
__global__ void EMAWeightCUDAFwdKernel(int64_t N, int64_t L,
                                       const T* __restrict__ p,
                                       const c10::complex<T>* __restrict__ v,
                                       T* __restrict__ w) {
  __shared__ T p_shared[cuda_utils::kWarpSize];

  const int64_t i = blockIdx.x;
  if (threadIdx.x < N) {
    p_shared[threadIdx.x] = p[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum = T(0);
    for (int64_t k = 0; k < N; ++k) {
      sum += (p_shared[k] * v[(i * N + k) * (L + 1) + j]).real();
    }
    w[i * L + j] = sum;
  }
}

template <typename T>
__global__ void EMAParametersBatchSize1CUDAFwdKernel(
    int64_t N, int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ h, T* __restrict__ weight,
    T* __restrict__ bias) {
  __shared__ T q_shared[cuda_utils::kWarpSize * 2];
  __shared__ T u_shared[cuda_utils::kWarpSize * 2];  // p * gamma
  __shared__ T v_shared[cuda_utils::kWarpSize * 2];  // q * gamma * h
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(q_shared);
  c10::complex<T>* u_ptr = reinterpret_cast<c10::complex<T>*>(u_shared);
  c10::complex<T>* v_ptr = reinterpret_cast<c10::complex<T>*>(v_shared);

  const int64_t i = blockIdx.x;
  T* w_ptr = weight + i * L;
  T* b_ptr = bias + i * L;

  if (threadIdx.x < N) {
    const c10::complex<T> q = log_q[i * N + threadIdx.x];
    const c10::complex<T> g = gamma[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = q;
    u_ptr[threadIdx.x] = p[i * N + threadIdx.x] * g;
    v_ptr[threadIdx.x] = complex_utils::Exp(q) * g * h[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum1 = T(0);
    T sum2 = T(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw =
          complex_utils::Exp(q_ptr[k] * static_cast<T>(j));
      sum1 += complex_utils::RealOfProduct(qw, u_ptr[k]);
      sum2 += complex_utils::RealOfProduct(qw, v_ptr[k]);
    }
    w_ptr[j] = sum1;
    b_ptr[j] = sum2;
  }
}

template <typename T, int64_t N>
__global__ void EMAParametersBatchSize1CUDAFwdKernel(
    int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ h, T* __restrict__ weight,
    T* __restrict__ bias) {
  extern __shared__ float shm[];
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(shm);
  c10::complex<T>* u_ptr = q_ptr + N;  // p * gamma
  c10::complex<T>* v_ptr = u_ptr + N;  // q * gamma * h
  c10::complex<T>* q_pow = v_ptr + N;

  const int64_t inner_size = blockDim.x + 1;
  const int64_t i = blockIdx.x;
  T* w_ptr = weight + i * L;
  T* b_ptr = bias + i * L;

  if (threadIdx.x < N) {
    const c10::complex<T> q = log_q[i * N + threadIdx.x];
    const c10::complex<T> g = gamma[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = q;
    u_ptr[threadIdx.x] = p[i * N + threadIdx.x] * g;
    v_ptr[threadIdx.x] = complex_utils::Exp(q) * g * h[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * inner_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T>(threadIdx.x));
  }
  if (threadIdx.x < N) {
    q_pow[threadIdx.x * inner_size + blockDim.x] =
        complex_utils::Exp(q_ptr[threadIdx.x] * static_cast<T>(blockDim.x));
  }
  __syncthreads();

  c10::complex<T> c[N];
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    c[k] = c10::complex<T>(1);
  }

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum1 = T(0);
    T sum2 = T(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw = q_pow[k * inner_size + threadIdx.x] * c[k];
      sum1 += complex_utils::RealOfProduct(qw, u_ptr[k]);
      sum2 += complex_utils::RealOfProduct(qw, v_ptr[k]);
      c[k] *= q_pow[k * inner_size + blockDim.x];
    }
    w_ptr[j] = sum1;
    b_ptr[j] = sum2;
  }
}

template <typename T, int64_t B>
__global__ void EMABiasCUDAFwdKernel(int64_t D, int64_t N, int64_t L,
                                     const c10::complex<T>* __restrict__ log_q,
                                     const c10::complex<T>* __restrict__ gamma,
                                     const c10::complex<T>* __restrict__ h,
                                     T* __restrict__ bias) {
  __shared__ T q_shared[cuda_utils::kWarpSize * 2];
  __shared__ T r_shared[cuda_utils::kWarpSize * 2];
  __shared__ T c_shared[B * cuda_utils::kWarpSize * 2];  // gamma * h
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(q_shared);
  c10::complex<T>* r_ptr = reinterpret_cast<c10::complex<T>*>(r_shared);
  c10::complex<T>* c_ptr = reinterpret_cast<c10::complex<T>*>(c_shared);

  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    r_ptr[threadIdx.x] = gamma[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t b = j / N;
    const int64_t k = j % N;
    c_ptr[j] = r_ptr[k] * h[(b * D + i) * N + k];
  }
  __syncthreads();

  T sum[B];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      sum[b] = T(0);
    }
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw =
          complex_utils::Exp(q_ptr[k] * static_cast<T>(j + 1));
#pragma unroll
      for (int64_t b = 0; b < B; ++b) {
        sum[b] += complex_utils::RealOfProduct(qw, c_ptr[b * N + k]);
      }
    }
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      bias[(b * D + i) * L + j] = sum[b];
    }
  }
}

template <typename T, int64_t B, int64_t N>
__global__ void EMABiasCUDAFwdKernel(int64_t D, int64_t L,
                                     const c10::complex<T>* __restrict__ log_q,
                                     const c10::complex<T>* __restrict__ gamma,
                                     const c10::complex<T>* __restrict__ h,
                                     T* __restrict__ bias) {
  extern __shared__ float shm[];
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(shm);
  c10::complex<T>* r_ptr = q_ptr + N;
  c10::complex<T>* c_ptr = r_ptr + N;
  c10::complex<T>* q_pow = c_ptr + B * N;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    r_ptr[threadIdx.x] = gamma[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t b = j / N;
    const int64_t k = j % N;
    c_ptr[j] = r_ptr[k] * h[(b * D + i) * N + k];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * blockDim.x + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T>(threadIdx.x + 1));
  }
  __syncthreads();

  c10::complex<T> c[N];
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    c[k] = c10::complex<T>(1);
  }

  T sum[B];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      sum[b] = T(0);
    }
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw = q_pow[k * blockDim.x + threadIdx.x] * c[k];
#pragma unroll
      for (int64_t b = 0; b < B; ++b) {
        sum[b] += complex_utils::RealOfProduct(qw, c_ptr[b * N + k]);
      }
      c[k] *= q_pow[k * blockDim.x + blockDim.x - 1];
    }
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      bias[(b * D + i) * L + j] = sum[b];
    }
  }
}

template <typename T>
__global__ void EMAWeightCUDABwdKernel(
    int64_t N, int64_t L, const T* __restrict__ w_grad, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ p_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const T p_v = p[i * N + j];
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = complex_utils::Exp(log_q_v);

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const T dw = w_grad[i * L + k];
    const c10::complex<T> qw1 =
        k == 0 ? c10::complex<T>(0)
               : complex_utils::Exp(log_q_v * static_cast<T>(k - 1));
    const c10::complex<T> qw2 = k == 0 ? c10::complex<T>(1) : qw1 * q_v;
    sum1 += qw1 * (dw * static_cast<T>(k));
    sum2 += qw2 * dw;
  }
  sum1 = reduce::BlockReduce(sum1, sum1_shared_ptr);
  sum2 = reduce::BlockReduce(sum2, sum2_shared_ptr);

  if (threadIdx.x == 0) {
    p_grad[i * N + j] = complex_utils::RealOfProduct(sum2, gamma_v);
    q_grad[i * N + j] = std::conj(sum1 * p_v * gamma_v);
    gamma_grad[i * N + j] = std::conj(sum2 * p_v);
  }
}

template <typename T, int64_t N>
__global__ void EMAWeightCUDABwdKernel(
    int64_t L, const T* __restrict__ w_grad, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ p_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad) {
  extern __shared__ float shm[];
  T* p_ptr = reinterpret_cast<T*>(shm);
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(p_ptr + N);
  c10::complex<T>* r_ptr = q_ptr + N;
  c10::complex<T>* sum1_shared = r_ptr + N;
  c10::complex<T>* sum2_shared = sum1_shared + cuda_utils::kWarpSize;
  c10::complex<T>* q_pow = sum2_shared + cuda_utils::kWarpSize;

  const int64_t inner_size = blockDim.x + 2;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    p_ptr[threadIdx.x] = p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    r_ptr[threadIdx.x] = gamma[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * inner_size + threadIdx.x + 1] =
        complex_utils::Exp(q_ptr[k] * static_cast<T>(threadIdx.x));
  }
  if (threadIdx.x < N) {
    // TODO: Check inf here.
    q_pow[threadIdx.x * inner_size] = complex_utils::Exp(-q_ptr[threadIdx.x]);
    q_pow[threadIdx.x * inner_size + blockDim.x + 1] =
        complex_utils::Exp(q_ptr[threadIdx.x] * static_cast<T>(blockDim.x));
  }
  __syncthreads();

  c10::complex<T> c[N];
  c10::complex<T> sum1[N];
  c10::complex<T> sum2[N];
#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    c[k] = c10::complex<T>(T(1));
    sum1[k] = c10::complex<T>(T(0));
    sum2[k] = c10::complex<T>(T(0));
  }

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const T dw = w_grad[i * L + j];
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw1 =
          j == 0 ? c10::complex<T>(T(0))
                 : q_pow[k * inner_size + threadIdx.x] * c[k];
      const c10::complex<T> qw2 =
          q_pow[k * inner_size + threadIdx.x + 1] * c[k];
      sum1[k] += qw1 * (dw * static_cast<T>(j));
      sum2[k] += qw2 * dw;
      c[k] *= q_pow[k * inner_size + blockDim.x + 1];
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
      p_grad[i * N + k] = complex_utils::RealOfProduct(sum2[k], r_ptr[k]);
      q_grad[i * N + k] = std::conj(sum1[k] * p_ptr[k] * r_ptr[k]);
      gamma_grad[i * N + k] = std::conj(sum2[k] * p_ptr[k]);
    }
  }
}

template <typename T>
__global__ void EMAParametersBatchSize1CUDABwdKernel(
    int64_t N, int64_t L, const T* __restrict__ w_grad,
    const T* __restrict__ b_grad, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ h, T* __restrict__ p_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad, c10::complex<T>* h_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum3_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum4_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);
  c10::complex<T>* sum3_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum3_shared);
  c10::complex<T>* sum4_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum4_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const T* w_grad_ptr = w_grad + i * L;
  const T* b_grad_ptr = b_grad + i * L;
  const T p_v = p[i * N + j];
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = complex_utils::Exp(log_q_v);
  const c10::complex<T> h_v = h[i * N + j];

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  c10::complex<T> sum3(T(0));
  c10::complex<T> sum4(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const T dw = w_grad_ptr[k];
    const T db = b_grad_ptr[k];
    const c10::complex<T> qw1 =
        k == 0 ? c10::complex<T>(0)
               : complex_utils::Exp(log_q_v * static_cast<T>(k - 1));
    const c10::complex<T> qw2 = k == 0 ? c10::complex<T>(1) : qw1 * q_v;
    const c10::complex<T> qw3 = qw2 * q_v;
    sum1 += qw1 * (dw * static_cast<T>(k));
    sum2 += qw2 * dw;
    sum3 += qw2 * (db * static_cast<T>(k + 1));
    sum4 += qw3 * db;
  }
  sum1 = reduce::BlockReduce(sum1, sum1_shared_ptr);
  sum2 = reduce::BlockReduce(sum2, sum2_shared_ptr);
  sum3 = reduce::BlockReduce(sum3, sum3_shared_ptr);
  sum4 = reduce::BlockReduce(sum4, sum4_shared_ptr);

  if (threadIdx.x == 0) {
    p_grad[i * N + j] = complex_utils::RealOfProduct(sum2, gamma_v);
    q_grad[i * N + j] = std::conj((sum1 * p_v + sum3 * h_v) * gamma_v);
    gamma_grad[i * N + j] = std::conj(sum2 * p_v + sum4 * h_v);
    h_grad[i * N + j] = std::conj(sum4 * gamma_v);
  }
}

template <typename T, int64_t N>
__global__ void EMAParametersBatchSize1CUDABwdKernel(
    int64_t L, const T* __restrict__ w_grad, const T* __restrict__ b_grad,
    const T* __restrict__ p, const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ h, T* __restrict__ p_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad, c10::complex<T>* h_grad) {
  extern __shared__ float shm[];
  T* p_ptr = reinterpret_cast<T*>(shm);
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(p_ptr + N);
  c10::complex<T>* r_ptr = q_ptr + N;
  c10::complex<T>* h_ptr = r_ptr + N;
  c10::complex<T>* sum1_shared = h_ptr + N;
  c10::complex<T>* sum2_shared = sum1_shared + cuda_utils::kWarpSize;
  c10::complex<T>* q_pow = sum2_shared + cuda_utils::kWarpSize;

  const int64_t inner_size = blockDim.x + 2;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    p_ptr[threadIdx.x] = p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    r_ptr[threadIdx.x] = gamma[i * N + threadIdx.x];
    h_ptr[threadIdx.x] = h[i * N + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * inner_size + threadIdx.x + 1] =
        complex_utils::Exp(q_ptr[k] * static_cast<T>(threadIdx.x));
  }
  if (threadIdx.x < N) {
    // TODO: Check inf here.
    q_pow[threadIdx.x * inner_size] = complex_utils::Exp(-q_ptr[threadIdx.x]);
    q_pow[threadIdx.x * inner_size + blockDim.x + 1] =
        complex_utils::Exp(q_ptr[threadIdx.x] * static_cast<T>(blockDim.x));
  }
  __syncthreads();

  c10::complex<T> c[N];
  c10::complex<T> sum1[N];
  c10::complex<T> sum2[N];

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    c[k] = c10::complex<T>(T(1));
    sum1[k] = c10::complex<T>(T(0));
    sum2[k] = c10::complex<T>(T(0));
  }
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const T dw = w_grad[i * L + j];
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw1 =
          j == 0 ? c10::complex<T>(0)
                 : q_pow[k * inner_size + threadIdx.x] * c[k];
      const c10::complex<T> qw2 =
          q_pow[k * inner_size + threadIdx.x + 1] * c[k];
      sum1[k] += qw1 * (dw * static_cast<T>(j));
      sum2[k] += qw2 * dw;
      c[k] *= q_pow[k * inner_size + blockDim.x + 1];
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
      p_grad[i * N + k] = complex_utils::RealOfProduct(sum2[k], r_ptr[k]);
      q_grad[i * N + k] = std::conj(sum1[k] * p_ptr[k] * r_ptr[k]);
      gamma_grad[i * N + k] = std::conj(sum2[k] * p_ptr[k]);
    }
  }

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    c[k] = c10::complex<T>(T(1));
    sum1[k] = c10::complex<T>(T(0));
    sum2[k] = c10::complex<T>(T(0));
  }
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const T db = b_grad[i * L + j];
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw1 =
          q_pow[k * inner_size + threadIdx.x + 1] * c[k];
      const c10::complex<T> qw2 =
          q_pow[k * inner_size + threadIdx.x + 2] * c[k];
      sum1[k] += qw1 * (db * static_cast<T>(j + 1));
      sum2[k] += qw2 * db;
      c[k] *= q_pow[k * inner_size + blockDim.x + 1];
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
      q_grad[i * N + k] += std::conj(sum1[k] * h_ptr[k] * r_ptr[k]);
      gamma_grad[i * N + k] += std::conj(sum2[k] * h_ptr[k]);
      h_grad[i * N + k] = std::conj(sum2[k] * r_ptr[k]);
    }
  }
}

template <typename T, int64_t B>
__global__ void EMABiasCUDABwdKernel(int64_t D, int64_t N, int64_t L,
                                     const T* __restrict__ b_grad,
                                     const c10::complex<T>* __restrict__ log_q,
                                     const c10::complex<T>* __restrict__ gamma,
                                     const c10::complex<T>* __restrict__ h,
                                     c10::complex<T>* __restrict__ q_grad,
                                     c10::complex<T>* __restrict__ gamma_grad,
                                     c10::complex<T>* h_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  __shared__ T q_pow_data[(cuda_utils::kCUDAMaxNumThreadsPerBlock + 1) * 2];
  c10::complex<T>* q_pow = reinterpret_cast<c10::complex<T>*>(q_pow_data);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];

  q_pow[threadIdx.x] =
      complex_utils::Exp(log_q_v * static_cast<T>(threadIdx.x));
  if (threadIdx.x == 0) {
    q_pow[blockDim.x] =
        complex_utils::Exp(log_q_v * static_cast<T>(blockDim.x));
  }
  __syncthreads();

  c10::complex<T> sum1[B];
  c10::complex<T> sum2[B];
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
    sum1[b] = c10::complex<T>(0);
    sum2[b] = c10::complex<T>(0);
  }

  c10::complex<T> c(T(1));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const c10::complex<T> qw1 = q_pow[threadIdx.x] * c;
    const c10::complex<T> qw2 = q_pow[threadIdx.x + 1] * c;
    c *= q_pow[blockDim.x];
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      const T db = b_grad[(b * D + i) * L + k];
      sum1[b] += db * qw1 * static_cast<T>(k + 1);
      sum2[b] += db * qw2;
    }
  }
#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
    sum1[b] = reduce::BlockReduce(
        sum1[b], reinterpret_cast<c10::complex<T>*>(sum1_shared));
    sum2[b] = reduce::BlockReduce(
        sum2[b], reinterpret_cast<c10::complex<T>*>(sum2_shared));
  }

  if (threadIdx.x == 0) {
    c10::complex<T> dq(T(0));
    c10::complex<T> dr(T(0));
#pragma unroll
    for (int64_t b = 0; b < B; ++b) {
      const c10::complex<T> h_v = h[(b * D + i) * N + j];
      dq += sum1[b] * h_v;
      dr += sum2[b] * h_v;
      h_grad[(b * D + i) * N + j] = std::conj(sum2[b] * gamma_v);
    }
    q_grad[i * N + j] += std::conj(dq * gamma_v);
    gamma_grad[i * N + j] += std::conj(dr);
  }
}

template <typename T, int64_t B, int64_t N>
__global__ void EMABiasCUDABwdKernel(int64_t D, int64_t L,
                                     const T* __restrict__ b_grad,
                                     const c10::complex<T>* __restrict__ log_q,
                                     const c10::complex<T>* __restrict__ gamma,
                                     const c10::complex<T>* __restrict__ h,
                                     c10::complex<T>* __restrict__ q_grad,
                                     c10::complex<T>* __restrict__ gamma_grad,
                                     c10::complex<T>* h_grad) {
  extern __shared__ float shm[];
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(shm);
  c10::complex<T>* r_ptr = q_ptr + N;
  c10::complex<T>* h_ptr = r_ptr + N;
  c10::complex<T>* dq_ptr = h_ptr + B * N;
  c10::complex<T>* dr_ptr = dq_ptr + N;
  c10::complex<T>* sum1_shared = dr_ptr + N;
  c10::complex<T>* sum2_shared = sum1_shared + cuda_utils::kWarpSize;
  c10::complex<T>* q_pow = sum2_shared + cuda_utils::kWarpSize;

  const int64_t inner_size = blockDim.x + 1;
  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    r_ptr[threadIdx.x] = gamma[i * N + threadIdx.x];
    dq_ptr[threadIdx.x] = c10::complex<T>(0);
    dr_ptr[threadIdx.x] = c10::complex<T>(0);
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t b = j / N;
    const int64_t k = j % N;
    h_ptr[j] = h[(b * D + i) * N + k];
  }
  __syncthreads();

#pragma unroll
  for (int64_t k = 0; k < N; ++k) {
    q_pow[k * inner_size + threadIdx.x] =
        complex_utils::Exp(q_ptr[k] * static_cast<T>(threadIdx.x));
  }
  if (threadIdx.x < N) {
    q_pow[threadIdx.x * inner_size + blockDim.x] =
        complex_utils::Exp(q_ptr[threadIdx.x] * static_cast<T>(blockDim.x));
  }
  __syncthreads();

  c10::complex<T> c[N];
  c10::complex<T> sum1[N];
  c10::complex<T> sum2[N];

#pragma unroll
  for (int64_t b = 0; b < B; ++b) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      c[k] = c10::complex<T>(1);
      sum1[k] = c10::complex<T>(0);
      sum2[k] = c10::complex<T>(0);
    }

    for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
      const T db = b_grad[(b * D + i) * L + j];
#pragma unroll
      for (int64_t k = 0; k < N; ++k) {
        const c10::complex<T> qw1 = q_pow[k * inner_size + threadIdx.x] * c[k];
        const c10::complex<T> qw2 =
            q_pow[k * inner_size + threadIdx.x + 1] * c[k];
        c[k] *= q_pow[k * inner_size + blockDim.x];
        sum1[k] += db * qw1 * static_cast<T>(j + 1);
        sum2[k] += db * qw2;
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
        dq_ptr[k] += std::conj(sum1[k] * h_ptr[b * N + k] * r_ptr[k]);
        dr_ptr[k] += std::conj(sum2[k] * h_ptr[b * N + k]);
        h_grad[(b * D + i) * N + k] = std::conj(sum2[k] * r_ptr[k]);
      }
    }
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int64_t k = 0; k < N; ++k) {
      q_grad[i * N + k] += dq_ptr[k];
      gamma_grad[i * N + k] += dr_ptr[k];
    }
  }
}

template <typename T>
__global__ void EMAVandermondeCUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ v_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;

  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = complex_utils::Exp(log_q_v);
  const c10::complex<T>* v_grad_ptr = v_grad + (i * N + j) * L;

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const c10::complex<T> dv = v_grad_ptr[k];
    const c10::complex<T> qw1 = complex_utils::Exp(log_q_v * static_cast<T>(k));
    const c10::complex<T> qw2 = qw1 * q_v;
    sum1 += dv * qw1 * static_cast<T>(k + 1);
    sum2 += dv * qw2;
  }
  sum1 = reduce::BlockReduce(sum1, sum1_shared_ptr);
  sum2 = reduce::BlockReduce(sum2, sum2_shared_ptr);

  if (threadIdx.x == 0) {
    q_grad[i * N + j] += std::conj(sum1 * gamma_v);
    gamma_grad[i * N + j] += std::conj(sum2);
  }
}

#define DISPATCH_EMA_WEIGHT_CUDA_FWD_KERNEL(                                \
    KernelFunc, T, num_threads, shm_size, cuda_stream, D, N, L, ...)        \
  do {                                                                      \
    if (N == 16) {                                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, 16>, D, num_threads, shm_size, \
                               cuda_stream, L, __VA_ARGS__);                \
    } else {                                                                \
      KernelFunc<T><<<D, num_threads, 0, cuda_stream>>>(N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    }                                                                       \
  } while (false)

#define DISPATCH_EMA_WEIGHT_CUDA_BWD_KERNEL(                                \
    KernelFunc, T, num_threads, shm_size, cuda_stream, D, N, L, ...)        \
  do {                                                                      \
    if (N == 16) {                                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, 16>, D, num_threads, shm_size, \
                               cuda_stream, L, __VA_ARGS__);                \
    } else {                                                                \
      KernelFunc<T>                                                         \
          <<<dim3(D, N), num_threads, 0, cuda_stream>>>(N, L, __VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    }                                                                       \
  } while (false)

#define DISPATCH_EMA_BIAS_CUDA_FWD_KERNEL(                                  \
    KernelFunc, T, num_threads, shm_size, cuda_stream, B, D, N, L, ...)     \
  do {                                                                      \
    if (B == 2) {                                                           \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 2, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 2>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else if (B == 3) {                                                    \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 3, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 3>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else if (B == 4) {                                                    \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 4, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 4>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else if (B == 5) {                                                    \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 5, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 5>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else if (B == 6) {                                                    \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 6, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 6>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else if (B == 7) {                                                    \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 7, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 7>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else if (B == 8) {                                                    \
      if (N == 16) {                                                        \
        cuda_utils::LaunchKernel(KernelFunc<T, 8, 16>, D, num_threads,      \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__); \
      } else {                                                              \
        KernelFunc<T, 8>                                                    \
            <<<D, num_threads, 0, cuda_stream>>>(D, N, L, __VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                     \
      }                                                                     \
    } else {                                                                \
      TORCH_CHECK(false);                                                   \
    }                                                                       \
  } while (false)

#define DISPATCH_EMA_BIAS_CUDA_BWD_KERNEL(                                     \
    KernelFunc, T, num_threads, shm_size, cuda_stream, B, D, N, L, ...)        \
  do {                                                                         \
    if (B == 2) {                                                              \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 2, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 2>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else if (B == 3) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 3, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 3>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else if (B == 4) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 4, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 4>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else if (B == 5) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 5, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 5>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else if (B == 6) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 6, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 6>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else if (B == 7) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 7, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 7>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else if (B == 8) {                                                       \
      if (N == 16) {                                                           \
        cuda_utils::LaunchKernel(KernelFunc<T, 8, 16>, D, num_threads,         \
                                 shm_size, cuda_stream, D, L, __VA_ARGS__);    \
      } else {                                                                 \
        cuda_utils::LaunchKernel(KernelFunc<T, 8>, dim3(D, N), num_threads, 0, \
                                 cuda_stream, D, N, L, __VA_ARGS__);           \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false);                                                      \
    }                                                                          \
  } while (false)

template <typename T>
void EMAParametersCUDAFwdImpl(const torch::Tensor& p,
                              const torch::Tensor& log_q,
                              const torch::Tensor& gamma,
                              const torch::Tensor& h, int64_t L,
                              torch::Tensor& w, c10::optional<torch::Tensor>& b,
                              c10::optional<torch::Tensor>& v) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);
  TORCH_CHECK(N <= cuda_utils::kWarpSize);

  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* w_data = w.data_ptr<T>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(p));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);

  if (!h.defined()) {
    const int64_t shm_size =
        (2 * N + N * (num_threads + 1)) * sizeof(c10::complex<T>);
    DISPATCH_EMA_WEIGHT_CUDA_FWD_KERNEL(EMAWeightCUDAFwdKernel, T, num_threads,
                                        shm_size, cuda_stream, D, N, L, p_data,
                                        log_q_data, gamma_data, w_data);
    return;
  }

  const int64_t B = h.size(0);
  const c10::complex<T>* h_data = h.data_ptr<c10::complex<T>>();

  if (B == 1) {
    b = c10::make_optional(torch::empty({B, D, L}, p.options()));
    T* b_data = b->data_ptr<T>();
    const int64_t shm_size =
        (3 * N + N * (num_threads + 1)) * sizeof(c10::complex<T>);
    DISPATCH_EMA_WEIGHT_CUDA_FWD_KERNEL(EMAParametersBatchSize1CUDAFwdKernel, T,
                                        num_threads, shm_size, cuda_stream, D,
                                        N, L, p_data, log_q_data, gamma_data,
                                        h_data, w_data, b_data);
    return;
  }

  if (B <= 8) {
    b = c10::make_optional(torch::empty({B, D, L}, p.options()));
    T* b_data = b->data_ptr<T>();

    const int64_t w_shm_size =
        (2 * N + N * (num_threads + 1)) * sizeof(c10::complex<T>);
    DISPATCH_EMA_WEIGHT_CUDA_FWD_KERNEL(EMAWeightCUDAFwdKernel, T, num_threads,
                                        w_shm_size, cuda_stream, D, N, L,
                                        p_data, log_q_data, gamma_data, w_data);

    const int64_t b_shm_size =
        (2 * N + B * N + N * num_threads) * sizeof(c10::complex<T>);
    DISPATCH_EMA_BIAS_CUDA_FWD_KERNEL(EMABiasCUDAFwdKernel, T, num_threads,
                                      b_shm_size, cuda_stream, B, D, N, L,
                                      log_q_data, gamma_data, h_data, b_data);
    return;
  }

  v = c10::make_optional(torch::empty({D, N, L + 1}, log_q.options()));
  c10::complex<T>* v_data = v->data_ptr<c10::complex<T>>();

  EMAVandermondeCUDAFwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
      N, L, log_q_data, gamma_data, v_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  EMAWeightCUDAFwdKernel<T>
      <<<D, num_threads, 0, cuda_stream>>>(N, L, p_data, v_data, w_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor y = torch::empty({B, D, L}, log_q.options());
  c10::complex<T>* y_data = y.data_ptr<c10::complex<T>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T> kAlpha(1);
  const c10::complex<T> kBeta(0);
  blas::GemmStridedBatchedCUDA<c10::complex<T>>(
      handle, blas::TransposeOp::kN, blas::TransposeOp::kN, D, B, L, N, kAlpha,
      h_data, D * N, N, v_data + 1, L + 1, N * (L + 1), kBeta, y_data, D * L,
      L);
  b = c10::make_optional(torch::real(y));
  // b = c10::make_optional(torch::real(y).contiguous());
}

template <typename T>
void EMAParametersCUDABwdImpl(
    const torch::Tensor& w_grad, const torch::Tensor& b_grad,
    const torch::Tensor& p, const torch::Tensor& log_q,
    const torch::Tensor& gamma, const torch::Tensor& h, const torch::Tensor& v,
    torch::Tensor& p_grad, torch::Tensor& q_grad, torch::Tensor& gamma_grad,
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

  at::cuda::OptionalCUDAGuard guard(at::device_of(p));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = cuda_utils::RowwiseNumThreads(L, kMaxNumThreads);

  if (!h.defined()) {
    const int64_t shm_size =
        N * sizeof(T) +
        (2 * N + 2 * cuda_utils::kWarpSize + N * (num_threads + 2)) *
            sizeof(c10::complex<T>);
    DISPATCH_EMA_WEIGHT_CUDA_BWD_KERNEL(
        EMAWeightCUDABwdKernel, T, num_threads, shm_size, cuda_stream, D, N, L,
        w_grad_data, p_data, log_q_data, gamma_data, p_grad_data, q_grad_data,
        gamma_grad_data);
    return;
  }

  TORCH_CHECK(b_grad.defined());
  h_grad = c10::make_optional(torch::empty_like(h));
  const int64_t B = h.size(0);
  const T* b_grad_data = b_grad.data_ptr<T>();
  const c10::complex<T>* h_data = h.data_ptr<c10::complex<T>>();
  c10::complex<T>* h_grad_data = h_grad->data_ptr<c10::complex<T>>();

  if (B == 1) {
    const int64_t shm_size =
        N * sizeof(T) +
        (3 * N + 2 * cuda_utils::kWarpSize + N * (num_threads + 2)) *
            sizeof(c10::complex<T>);
    DISPATCH_EMA_WEIGHT_CUDA_BWD_KERNEL(
        EMAParametersBatchSize1CUDABwdKernel, T, num_threads, shm_size,
        cuda_stream, D, N, L, w_grad_data, b_grad_data, p_data, log_q_data,
        gamma_data, h_data, p_grad_data, q_grad_data, gamma_grad_data,
        h_grad_data);
    return;
  }

  const int64_t shm_size = N * sizeof(T) + (2 * N + 2 * cuda_utils::kWarpSize +
                                            N * (num_threads + 2)) *
                                               sizeof(c10::complex<T>);
  DISPATCH_EMA_WEIGHT_CUDA_BWD_KERNEL(
      EMAWeightCUDABwdKernel, T, num_threads, shm_size, cuda_stream, D, N, L,
      w_grad_data, p_data, log_q_data, gamma_data, p_grad_data, q_grad_data,
      gamma_grad_data);

  if (B <= 8) {
    const int64_t shm_size =
        (4 * N + B * N + 2 * cuda_utils::kWarpSize + N * (num_threads + 1)) *
        sizeof(c10::complex<T>);
    DISPATCH_EMA_BIAS_CUDA_BWD_KERNEL(
        EMABiasCUDABwdKernel, T, num_threads, shm_size, cuda_stream, B, D, N, L,
        b_grad_data, log_q_data, gamma_data, h_data, q_grad_data,
        gamma_grad_data, h_grad_data);
    return;
  }

  TORCH_CHECK(v.defined());
  torch::Tensor b_grad_complex =
      b_grad.to(c10::toComplexType(b_grad.scalar_type()));
  torch::Tensor v_grad = torch::empty({D, N, L}, v.options());

  const c10::complex<T>* b_grad_complex_data =
      b_grad_complex.data_ptr<c10::complex<T>>();
  const c10::complex<T>* v_data = v.data_ptr<c10::complex<T>>();
  c10::complex<T>* v_grad_data = v_grad.data_ptr<c10::complex<T>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T> kAlpha(1);
  const c10::complex<T> kBeta(0);
  blas::GemmStridedBatchedCUDA<c10::complex<T>>(
      handle, blas::TransposeOp::kN, blas::TransposeOp::kC, D, B, N, L, kAlpha,
      b_grad_complex_data, D * L, L, v_data + 1, L + 1, N * (L + 1), kBeta,
      h_grad_data, D * N, N);
  blas::GemmStridedBatchedCUDA<c10::complex<T>>(
      handle, blas::TransposeOp::kT, blas::TransposeOp::kN, D, N, L, B, kAlpha,
      h_data, D * N, N, b_grad_complex_data, D * L, L, kBeta, v_grad_data, L,
      N * L);
  EMAVandermondeCUDABwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
      N, L, log_q_data, gamma_data, v_grad_data, q_grad_data, gamma_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef DISPATCH_EMA_WEIGHT_CUDA_FWD_KERNEL
#undef DISPATCH_EMA_WEIGHT_CUDA_BWD_KERNEL

#undef DISPATCH_EMA_BIAS_CUDA_FWD_KERNEL
#undef DISPATCH_EMA_BIAS_CUDA_BWD_KERNEL

}  // namespace

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersCUDAFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& h, int64_t L) {
  const int64_t D = p.size(0);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor w = torch::empty(
      {D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> b = c10::nullopt;
  c10::optional<torch::Tensor> v = c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAParametersCUDAFwd", [&]() {
    EMAParametersCUDAFwdImpl<scalar_t>(
        *(p.expect_contiguous()), *(log_q.expect_contiguous()),
        *(gamma.expect_contiguous()), *(h_maybe_owned->expect_contiguous()), L,
        w, b, v);
  });

  return std::make_tuple<torch::Tensor, c10::optional<torch::Tensor>,
                         c10::optional<torch::Tensor>>(
      std::move(w), std::move(b), std::move(v));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersCUDABwd(const torch::Tensor& w_grad,
                     const c10::optional<torch::Tensor>& b_grad,
                     const torch::Tensor& p, const torch::Tensor& log_q,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& h,
                     const c10::optional<torch::Tensor>& v) {
  c10::MaybeOwned<torch::Tensor> b_grad_maybe_owned =
      at::borrow_from_optional_tensor(b_grad);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  c10::MaybeOwned<torch::Tensor> v_maybe_owned =
      at::borrow_from_optional_tensor(v);
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> h_grad = c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAParametersCUDABwd", [&]() {
    EMAParametersCUDABwdImpl<scalar_t>(
        *(w_grad.expect_contiguous()),
        *(b_grad_maybe_owned->expect_contiguous()), *(p.expect_contiguous()),
        *(log_q.expect_contiguous()), *(gamma.expect_contiguous()),
        *(h_maybe_owned->expect_contiguous()),
        *(v_maybe_owned->expect_contiguous()), p_grad, q_grad, gamma_grad,
        h_grad);
  });

  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>>(
      std::move(p_grad), std::move(q_grad), std::move(gamma_grad),
      std::move(h_grad));
}

}  // namespace ops
}  // namespace megalodon
