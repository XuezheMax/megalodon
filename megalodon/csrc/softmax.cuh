#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <thrust/pair.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <cmath>
#include <limits>
#include <type_traits>

#include "cuda_utils.cuh"
#include "random_utils.cuh"
#include "reduce.cuh"
#include "register_utils.cuh"

namespace megalodon {
namespace softmax {

constexpr int64_t kMaxSoftmaxSize = 8192;

template <typename T, int64_t kElementsPerThread>
__inline__ __device__ void BlockFillImpl(int64_t size, T val, T* data) {
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx < size) {
      data[idx] = val;
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void BlockFillFP16x2(int64_t size, at::Half val,
                                           at::Half* data) {
  const float v = static_cast<float>(val);
  const __half2 val2 = __floats2half2_rn(v, v);
  __half2* data2 = reinterpret_cast<__half2*>(data);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx * 2 < size) {
      data2[idx] = val2;
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void BlockFillBF16x2(int64_t size, at::BFloat16 val,
                                           at::BFloat16* data) {
  const float v = static_cast<float>(val);
  const __nv_bfloat162 val2 = __floats2bfloat162_rn(v, v);
  __nv_bfloat162* data2 = reinterpret_cast<__nv_bfloat162*>(data);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx * 2 < size) {
      data2[idx] = val2;
    }
  }
}

template <typename T, int64_t kElementsPerThread>
__inline__ __device__ void BlockFill(int64_t size, T val, T* data) {
  if constexpr (std::is_same<T, at::Half>::value) {
    if (kElementsPerThread % 2 == 0 &&
        reinterpret_cast<std::uintptr_t>(data) % alignof(__half2) == 0 &&
        size % 2 == 0) {
      BlockFillFP16x2<kElementsPerThread>(size, val, data);
    } else {
      BlockFillImpl<at::Half, kElementsPerThread>(size, val, data);
    }
  } else if constexpr (std::is_same<T, at::BFloat16>::value) {
    if (kElementsPerThread % 2 == 0 &&
        reinterpret_cast<std::uintptr_t>(data) % alignof(__nv_bfloat162) == 0 &&
        size % 2 == 0) {
      BlockFillBF16x2<kElementsPerThread>(size, val, data);
    } else {
      BlockFillImpl<at::BFloat16, kElementsPerThread>(size, val, data);
    }
  } else {
    BlockFillImpl<T, kElementsPerThread>(size, val, data);
  }
}

template <typename T, typename T_ACC, int64_t kCapacity, int64_t kNumThreads>
__global__ void AttentionSoftmaxFwdKernel(int64_t outer_size,
                                          int64_t inner_size,
                                          bool use_causal_mask, const T* x,
                                          T* y) {
  constexpr int64_t kElementsPerThread = kCapacity / kNumThreads;
  constexpr T_ACC kInf = std::numeric_limits<T_ACC>::infinity();

  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  T_ACC x_acc[kElementsPerThread];

  const int64_t i = blockIdx.y * outer_size + blockIdx.x;
  const int64_t r = blockIdx.x;
  const int64_t n =
      use_causal_mask ? inner_size - outer_size + r + 1 : inner_size;
  T_ACC d = T_ACC(0);
  T_ACC m = -kInf;  // attn mask is -inf.

  register_utils::Load<T, T_ACC, kElementsPerThread>(x + i * inner_size, n,
                                                     inner_size, -kInf, x_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    m = c10::cuda::compat::max(m, x_acc[j]);
  }
  m = reduce::BlockAllReduce(m, shm, reduce::MaxOp<T_ACC>());
  if (std::isinf(m)) {
    BlockFill<T, kElementsPerThread>(inner_size, T(0), y + i * inner_size);
    return;
  }

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const T_ACC v = c10::cuda::compat::exp(x_acc[j] - m);
    x_acc[j] = v;
    d += v;
  }
  d = reduce::BlockAllReduce(d, shm, reduce::SumOp<T_ACC>());

  const T_ACC c = d == T_ACC(0) ? T_ACC(0) : T_ACC(1) / d;
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    x_acc[j] *= c;
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(x_acc, inner_size,
                                                     y + i * inner_size);
}

// DropKey
// https://openaccess.thecvf.com/content/CVPR2023/html/Li_DropKey_for_Vision_Transformer_CVPR_2023_paper.html
template <typename T, typename T_ACC, int64_t kCapacity, int64_t kNumThreads>
__global__ void AttentionDropKeySoftmaxFwdKernel(
    at::PhiloxCudaState philox_args, int64_t outer_size, int64_t inner_size,
    T_ACC dropout, bool use_causal_mask, const T* x, T* y) {
  constexpr int64_t kElementsPerThread = kCapacity / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  constexpr T_ACC kInf = std::numeric_limits<T_ACC>::infinity();

  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  T_ACC x_acc[kCapacityPerThread];

  const int64_t i = blockIdx.y * outer_size + blockIdx.x;
  const int64_t r = blockIdx.x;
  const int64_t n =
      use_causal_mask ? inner_size - outer_size + r + 1 : inner_size;
  T_ACC d = T_ACC(0);
  T_ACC m = -kInf;  // attn mask is -inf.

  register_utils::Load<T, T_ACC, kElementsPerThread>(x + i * inner_size, n,
                                                     inner_size, -kInf, x_acc);

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(seed, i * blockDim.x + threadIdx.x, offset, &state);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    x_acc[j + 0] = rand4.x < dropout ? -kInf : x_acc[j + 0];
    x_acc[j + 1] = rand4.y < dropout ? -kInf : x_acc[j + 1];
    x_acc[j + 2] = rand4.z < dropout ? -kInf : x_acc[j + 2];
    x_acc[j + 3] = rand4.w < dropout ? -kInf : x_acc[j + 3];
  }

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    m = c10::cuda::compat::max(m, x_acc[j]);
  }
  m = reduce::BlockAllReduce(m, shm, reduce::MaxOp<T_ACC>());
  if (std::isinf(m)) {
    BlockFill<T, kElementsPerThread>(inner_size, T(0), y + i * inner_size);
    return;
  }

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const T_ACC x = x_acc[j];
    const T_ACC v = c10::cuda::compat::exp(x_acc[j] - m);
    x_acc[j] = v;
    d += v;
  }
  d = reduce::BlockAllReduce(d, shm, reduce::SumOp<T_ACC>());

  const T_ACC c = d == T_ACC(0) ? T_ACC(0) : T_ACC(1) / d;
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    x_acc[j] *= c;
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(x_acc, inner_size,
                                                     y + i * inner_size);
}

template <typename T, typename T_ACC, int64_t kCapacity, int64_t kNumThreads>
__global__ void AttentionSoftmaxBwdKernel(int64_t outer_size,
                                          int64_t inner_size,
                                          bool use_causal_mask, const T* y_grad,
                                          const T* __restrict__ y, T* x_grad) {
  constexpr int kElementsPerThread = kCapacity / kNumThreads;

  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  T_ACC p_acc[kElementsPerThread];
  T_ACC o_acc[kElementsPerThread];

  const int64_t i = blockIdx.y * outer_size + blockIdx.x;
  const int64_t r = blockIdx.x;
  const int64_t n =
      use_causal_mask ? inner_size - outer_size + r + 1 : inner_size;
  T_ACC sum = T_ACC(0);

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * inner_size, n, inner_size, T_ACC(0), o_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y + i * inner_size, n, inner_size, T_ACC(0), p_acc);

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const T_ACC o = p_acc[j] * o_acc[j];
    o_acc[j] = o;
    sum += o;
  }
  sum = reduce::BlockAllReduce(sum, shm, reduce::SumOp<T_ACC>());

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    o_acc[j] -= p_acc[j] * sum;
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(o_acc, inner_size,
                                                     x_grad + i * inner_size);
}

}  // namespace softmax

#define DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(                               \
    KernelFunc, T, T_ACC, block_size, inner_size, shm_size, cuda_stream, ...) \
  do {                                                                        \
    if (inner_size <= 32) {                                                   \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 32, 32>, block_size, 32,  \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 64) {                                            \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 64, 32>, block_size, 32,  \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 128) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 128, 32>, block_size, 32, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 256) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32>, block_size, 32, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 512) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64>, block_size, 64, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 1024) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128>, block_size,   \
                               128, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 2048) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256>, block_size,   \
                               256, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 4096) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512>, block_size,   \
                               512, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 8192) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8192, 1024>, block_size,  \
                               1024, shm_size, cuda_stream, __VA_ARGS__);     \
    } else {                                                                  \
      TORCH_CHECK(false);                                                     \
    }                                                                         \
  } while (false)

}  // namespace megalodon
