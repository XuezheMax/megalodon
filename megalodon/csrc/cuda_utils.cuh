#pragma once

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <cstdint>
#include <limits>

#include "welford.h"

namespace megalodon {
namespace cuda_utils {

constexpr int64_t kWarpSize = 32;
constexpr int64_t kCUDANumThreads = 128;
constexpr int64_t kCUDABlockReduceNumThreads = 512;
constexpr int64_t kCUDAMaxNumThreadsPerBlock = 1024;
constexpr int64_t kColwiseThreshold = 256;
constexpr int64_t kMaxStaticSharedMemorySize = 49152;

struct __align__(8) BF16x4 {
  __nv_bfloat16 x0;
  __nv_bfloat16 x1;
  __nv_bfloat16 x2;
  __nv_bfloat16 x3;
};

struct __align__(16) BF16x8 {
  __nv_bfloat16 x0;
  __nv_bfloat16 x1;
  __nv_bfloat16 x2;
  __nv_bfloat16 x3;
  __nv_bfloat16 x4;
  __nv_bfloat16 x5;
  __nv_bfloat16 x6;
  __nv_bfloat16 x7;
};

template <typename T>
C10_HOST_DEVICE T Square(T x) {
  return x * x;
}

template <typename T>
C10_HOST_DEVICE T Cube(T x) {
  return x * x * x;
}

template <typename T>
__inline__ __device__ T WarpShflDown(T value, unsigned int delta,
                                     int width = warpSize,
                                     unsigned int mask = 0xFFFFFFFFU) {
  return WARP_SHFL_DOWN(value, delta, width, mask);
}

template <typename T>
__inline__ __device__ utils::WelfordData<T> WarpShflDown(
    utils::WelfordData<T> value, unsigned int delta, int width = warpSize,
    unsigned int mask = 0xFFFFFFFFU) {
  return {WARP_SHFL_DOWN(value.m0, delta, width, mask),
          WARP_SHFL_DOWN(value.m1, delta, width, mask),
          WARP_SHFL_DOWN(value.m2, delta, width, mask)};
}

template <typename T>
__inline__ __device__ T WarpShflXor(T value, unsigned int delta,
                                    int width = warpSize,
                                    unsigned int mask = 0xFFFFFFFFU) {
  return WARP_SHFL_XOR(value, delta, width, mask);
}

template <typename T>
__inline__ __device__ c10::complex<T> WarpShflXor(
    c10::complex<T> value, unsigned int delta, int width = warpSize,
    unsigned int mask = 0xFFFFFFFFU) {
  return {WARP_SHFL_XOR(value.real(), delta, width, mask),
          WARP_SHFL_XOR(value.imag(), delta, width, mask)};
}

template <typename T>
__inline__ __device__ utils::WelfordData<T> WarpShflXor(
    utils::WelfordData<T> value, unsigned int delta, int width = warpSize,
    unsigned int mask = 0xFFFFFFFFU) {
  return {WARP_SHFL_XOR(value.m0, delta, width, mask),
          WARP_SHFL_XOR(value.m1, delta, width, mask),
          WARP_SHFL_XOR(value.m2, delta, width, mask)};
}

constexpr int64_t RowwiseNumThreads(
    int64_t size, int64_t max_num_threads = kCUDAMaxNumThreadsPerBlock) {
  if (size >= 1024) {
    return std::min(int64_t(1024), max_num_threads);
  }
  if (size >= 512) {
    return std::min(int64_t(512), max_num_threads);
  }
  if (size >= 256) {
    return std::min(int64_t(256), max_num_threads);
  }
  if (size >= 128) {
    return std::min(int64_t(128), max_num_threads);
  }
  if (size >= 64) {
    return std::min(int64_t(64), max_num_threads);
  }
  return std::min(int64_t(32), max_num_threads);
}

template <class KernelFunc, class... Args>
void LaunchKernel(KernelFunc kernel, dim3 dg, dim3 db, int64_t shared_mem_size,
                  cudaStream_t cuda_stream, Args... args) {
  if (shared_mem_size > cuda_utils::kMaxStaticSharedMemorySize) {
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
  }
  kernel<<<dg, db, shared_mem_size, cuda_stream>>>(args...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_utils
}  // namespace megalodon
