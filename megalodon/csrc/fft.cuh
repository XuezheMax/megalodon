#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>
#include <thrust/swap.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "complex_utils.cuh"
#include "cuda_utils.cuh"
#include "twiddle_factor.cuh"

namespace megalodon {
namespace fft {

constexpr int kFFTMaxLength = 16384;
constexpr int kFFTNumThreads = 1024;

constexpr __device__ int Log2(int x) {
  switch (x) {
    case 1: {
      return 0;
    }
    case 2: {
      return 1;
    }
    case 4: {
      return 2;
    }
    case 8: {
      return 3;
    }
    case 16: {
      return 4;
    }
    case 32: {
      return 5;
    }
    case 64: {
      return 6;
    }
    case 128: {
      return 7;
    }
    case 256: {
      return 8;
    }
    case 512: {
      return 9;
    }
    case 1024: {
      return 10;
    }
    case 2048: {
      return 11;
    }
    case 4096: {
      return 12;
    }
    case 8192: {
      return 13;
    }
    case 16384: {
      return 14;
    }
    default: {
      return -1;
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void LoadAsComplexImpl(
    const T* __restrict__ src, int64_t size,
    c10::complex<T_ACC>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if constexpr (kFlip) {
      const int64_t rev = (__brev(kFFTSize - 1 - idx) >> (32 - kNumBits));
      const int64_t d = 2 * kFFTSize - size;
      const int64_t p = idx * 2;
      const int64_t q = idx * 2 + 1;
      const T_ACC x0 = p >= d ? static_cast<T_ACC>(src[p - d]) : T_ACC(0);
      const T_ACC x1 = q >= d ? static_cast<T_ACC>(src[q - d]) : T_ACC(0);
      dst[rev] = c10::complex<T_ACC>(x1, x0);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      const int64_t p = idx * 2;
      const int64_t q = idx * 2 + 1;
      const T_ACC x0 = p < size ? static_cast<T_ACC>(src[p]) : T_ACC(0);
      const T_ACC x1 = q < size ? static_cast<T_ACC>(src[q]) : T_ACC(0);
      dst[rev] = c10::complex<T_ACC>(x0, x1);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexFP32x2Impl(
    const float* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float2 kZero2 = {0.0f, 0.0f};
  const float2* src2 = reinterpret_cast<const float2*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = idx * 2 < size ? src2[idx] : kZero2;
    if constexpr (kFlip) {
      const int64_t rev = (__brev(size / 2 - 1 - idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.y, v2.x);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.x, v2.y);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexFP32x4Impl(
    const float* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float4 kZero4 = {0.0f, 0.0f, 0.0f, 0.0f};
  const float4* src4 = reinterpret_cast<const float4*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float4 v4 = idx * 4 < size ? src4[idx] : kZero4;
    if constexpr (kFlip) {
      const int64_t p = size / 2 - 1 - idx * 2;
      const int64_t q = size / 2 - 1 - idx * 2 - 1;
      const int64_t rev0 = (__brev(p) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(q) >> (32 - kNumBits));
      dst[rev0] = c10::complex<float>(v4.y, v4.x);
      dst[rev1] = c10::complex<float>(v4.w, v4.z);
    } else {
      const int64_t rev0 = (__brev(idx * 2 + 0) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(idx * 2 + 1) >> (32 - kNumBits));
      dst[rev0] = c10::complex<float>(v4.x, v4.y);
      dst[rev1] = c10::complex<float>(v4.z, v4.w);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexFP16x2Impl(
    const at::Half* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float2 kZero2 = {0.0f, 0.0f};
  const __half2* src2 = reinterpret_cast<const __half2*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = idx * 2 < size ? __half22float2(src2[idx]) : kZero2;
    if constexpr (kFlip) {
      const int64_t rev = (__brev(size / 2 - 1 - idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.y, v2.x);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.x, v2.y);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexBF16x2Impl(
    const at::BFloat16* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float2 kZero2 = {0.0f, 0.0f};
  const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const float2 v2 = idx * 2 < size ? __bfloat1622float2(src2[idx]) : kZero2;
#else
    const __nv_bfloat162 x2 = src2[idx];
    const float2 v2 = idx * 2 < size ? make_float2(__bfloat162float(x2.x),
                                                   __bfloat162float(x2.y))
                                     : kZero2;
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (kFlip) {
      const int64_t rev = (__brev(size / 2 - 1 - idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.y, v2.x);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.x, v2.y);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexBF16x4Impl(
    const at::BFloat16* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  const __nv_bfloat16 kZero = __float2bfloat16(0.0f);
  const cuda_utils::BF16x4 kZero4 = {kZero, kZero, kZero, kZero};
  const cuda_utils::BF16x4* src4 =
      reinterpret_cast<const cuda_utils::BF16x4*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const cuda_utils::BF16x4 v4 = idx * 4 < size ? src4[idx] : kZero4;
    if constexpr (kFlip) {
      const int64_t p = size / 2 - 1 - idx * 2;
      const int64_t q = size / 2 - 1 - idx * 2 - 1;
      const int64_t rev0 = (__brev(p) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(q) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v4.x1), __bfloat162float(v4.x0));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v4.x3), __bfloat162float(v4.x2));
    } else {
      const int64_t rev0 = (__brev(idx * 2 + 0) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(idx * 2 + 1) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v4.x0), __bfloat162float(v4.x1));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v4.x2), __bfloat162float(v4.x3));
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexBF16x8Impl(
    const at::BFloat16* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  const __nv_bfloat16 kZero = __float2bfloat16(0.0f);
  const cuda_utils::BF16x8 kZero8 = {kZero, kZero, kZero, kZero,
                                     kZero, kZero, kZero, kZero};
  const cuda_utils::BF16x8* src8 =
      reinterpret_cast<const cuda_utils::BF16x8*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 4; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const cuda_utils::BF16x8 v8 = idx * 8 < size ? src8[idx] : kZero8;
    if constexpr (kFlip) {
      const int64_t p = size / 2 - 1 - idx * 4;
      const int64_t q = size / 2 - 1 - idx * 4 - 1;
      const int64_t r = size / 2 - 1 - idx * 4 - 2;
      const int64_t s = size / 2 - 1 - idx * 4 - 3;
      const int64_t rev0 = (__brev(p) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(q) >> (32 - kNumBits));
      const int64_t rev2 = (__brev(r) >> (32 - kNumBits));
      const int64_t rev3 = (__brev(s) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v8.x1), __bfloat162float(v8.x0));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v8.x3), __bfloat162float(v8.x2));
      dst[rev2] =
          c10::complex<float>(__bfloat162float(v8.x5), __bfloat162float(v8.x4));
      dst[rev3] =
          c10::complex<float>(__bfloat162float(v8.x7), __bfloat162float(v8.x6));
    } else {
      const int64_t rev0 = (__brev(idx * 4 + 0) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(idx * 4 + 1) >> (32 - kNumBits));
      const int64_t rev2 = (__brev(idx * 4 + 2) >> (32 - kNumBits));
      const int64_t rev3 = (__brev(idx * 4 + 3) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v8.x0), __bfloat162float(v8.x1));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v8.x2), __bfloat162float(v8.x3));
      dst[rev2] =
          c10::complex<float>(__bfloat162float(v8.x4), __bfloat162float(v8.x5));
      dst[rev3] =
          c10::complex<float>(__bfloat162float(v8.x6), __bfloat162float(v8.x7));
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void LoadAsComplex(
    const T* __restrict__ src, int64_t size,
    c10::complex<T_ACC>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;

  if constexpr (std::is_same<T, float>::value &&
                std::is_same<T_ACC, float>::value) {
    if (kElementsPerThread % 2 == 0 &&
        reinterpret_cast<uintptr_t>(src) % sizeof(float4) == 0 &&
        size % 4 == 0) {
      LoadAsComplexFP32x4Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else if (reinterpret_cast<uintptr_t>(src) % sizeof(float2) == 0 &&
               size % 2 == 0) {
      LoadAsComplexFP32x2Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else {
      LoadAsComplexImpl<float, float, kFFTSize, kNumThreads, kFlip>(src, size,
                                                                    dst);
    }
  } else if constexpr (std::is_same<T, at::Half>::value &&
                       std::is_same<T_ACC, float>::value) {
    if (reinterpret_cast<uintptr_t>(src) % sizeof(__half2) == 0 &&
        size % 2 == 0) {
      LoadAsComplexFP16x2Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else {
      LoadAsComplexImpl<at::Half, float, kFFTSize, kNumThreads, kFlip>(
          src, size, dst);
    }
  } else if constexpr (std::is_same<T, at::BFloat16>::value &&
                       std::is_same<T_ACC, float>::value) {
    if (kElementsPerThread % 4 == 0 &&
        reinterpret_cast<uintptr_t>(src) % sizeof(cuda_utils::BF16x8) == 0 &&
        size % 8 == 0) {
      LoadAsComplexBF16x8Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else if (kElementsPerThread % 2 == 0 &&
               reinterpret_cast<uintptr_t>(src) % sizeof(cuda_utils::BF16x4) ==
                   0 &&
               size % 4 == 0) {
      LoadAsComplexBF16x4Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else if (reinterpret_cast<uintptr_t>(src) % sizeof(__nv_bfloat162) == 0 &&
               size % 2 == 0) {
      LoadAsComplexBF16x2Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else {
      LoadAsComplexImpl<at::BFloat16, float, kFFTSize, kNumThreads, kFlip>(
          src, size, dst);
    }
  } else {
    LoadAsComplexImpl<T, T_ACC, kFFTSize, kNumThreads, kFlip>(src, size, dst);
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void SaveAsRealImpl1(
    const c10::complex<T_ACC>* __restrict__ src, int64_t size, T_ACC scale,
    T* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  const T_ACC* src_real = reinterpret_cast<const T_ACC*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const int64_t p = idx * 2;
    const int64_t q = idx * 2 + 1;
    if constexpr (kFlip) {
      if (p < size) {
        dst[p] = static_cast<T>(src_real[size - 1 - p] * scale);
      }
      if (q < size) {
        dst[q] = static_cast<T>(src_real[size - 1 - q] * scale);
      }
    } else {
      if (p < size) {
        dst[p] = static_cast<T>(src_real[p] * scale);
      }
      if (q < size) {
        dst[q] = static_cast<T>(src_real[q] * scale);
      }
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void SaveAsRealImpl2(
    const c10::complex<T_ACC>* __restrict__ src, int64_t size, T_ACC scale,
    T* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  c10::complex<T>* dst_complex = reinterpret_cast<c10::complex<T>*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx * 2 < size) {
      if constexpr (kFlip) {
        const c10::complex<T_ACC> v = src[size / 2 - 1 - idx] * scale;
        dst_complex[idx] =
            c10::complex<T>(static_cast<T>(v.imag()), static_cast<T>(v.real()));
      } else if constexpr (std::is_same<T, T_ACC>::value) {
        dst_complex[idx] = src[idx] * scale;
      } else {
        const c10::complex<T_ACC> v = src[idx] * scale;
        dst_complex[idx] =
            c10::complex<T>(static_cast<T>(v.real()), static_cast<T>(v.imag()));
      }
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void SaveAsReal(
    const c10::complex<T_ACC>* __restrict__ src, int64_t size, T_ACC scale,
    T* __restrict__ dst) {
  if (reinterpret_cast<uintptr_t>(dst) % sizeof(c10::complex<T>) == 0 &&
      size % 2 == 0) {
    SaveAsRealImpl2<T, T_ACC, kFFTSize, kNumThreads, kFlip>(src, size, scale,
                                                            dst);
  } else {
    SaveAsRealImpl1<T, T_ACC, kFFTSize, kNumThreads, kFlip>(src, size, scale,
                                                            dst);
  }
}

template <typename T, int N, int kNumThreads>
__inline__ __device__ void BitReverse(const c10::complex<T>* src,
                                      c10::complex<T>* dst) {
  constexpr int kElementsPerThread = N / kNumThreads;
  constexpr int kNumBits = Log2(N);

  if (src == dst) {
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      const int idx = i * kNumThreads + threadIdx.x;
      const int rev = (__brev(idx) >> (32 - kNumBits));
      if (idx < rev) {
        thrust::swap(dst[idx], dst[rev]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      const int idx = i * kNumThreads + threadIdx.x;
      const int rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = src[idx];
    }
  }
}

template <typename T, bool kIFFT = false>
__inline__ __device__ c10::complex<T> WarpFFTImpl(c10::complex<T> x) {
#pragma unroll
  for (int offset = 1; offset < cuda_utils::kWarpSize; offset <<= 1) {
    const int r = (threadIdx.x & (offset - 1));
    const c10::complex<T> w = cuda_utils::TwiddleFactor<T>(offset, r);
    const c10::complex<T> y = cuda_utils::WarpShflXor(x, offset);
    // x = (threadIdx.x & offset) ? (y - x * (kIFFT ? w : std::conj(w)))
    //                            : (x + y * (kIFFT ? w : std::conj(w)));
    if constexpr (kIFFT) {
      x = (threadIdx.x & offset) ? (y - x * w) : (x + y * w);
    } else {
      x = (threadIdx.x & offset) ? (y - x * std::conj(w))
                                 : (x + y * std::conj(w));
    }
  }
  return x;
}

// Cooley–Tukey FFT
template <typename T, int N, bool kIFFT = false>
__inline__ __device__ void BlockFFTImpl(c10::complex<T>* shared_mem) {
  constexpr int D = Log2(cuda_utils::kWarpSize);
  constexpr int K = Log2(N);

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    shared_mem[i] = WarpFFTImpl<T, kIFFT>(shared_mem[i]);
  }
  // __syncthreads();

#pragma unroll
  for (int i = D; i < K; ++i) {
    __syncthreads();
    const int m = (1 << i);
    for (int j = threadIdx.x; j < N / 2; j += blockDim.x) {
      const int r = (j & (m - 1));
      const int p = ((j >> i) << (i + 1)) + r;
      const int q = p + m;
      const c10::complex<T> w = cuda_utils::TwiddleFactor<T>(m, r);
      const c10::complex<T> u = shared_mem[p];
      const c10::complex<T> v = shared_mem[q] * (kIFFT ? w : std::conj(w));
      shared_mem[p] = u + v;
      shared_mem[q] = u - v;
    }
    // __syncthreads();
  }
}

template <typename T, int N, int kNumThreads>
__inline__ __device__ void RFFTPostProcess(const c10::complex<T>* src,
                                           c10::complex<T>* dst) {
  constexpr int kElementsPerThread = N / kNumThreads;

  if (src == dst) {
    assert(kElementsPerThread > 1);
#pragma unroll
    for (int i = 0; i < kElementsPerThread / 2; ++i) {
      const int p = i * kNumThreads + threadIdx.x;
      const int q = p == 0 ? 0 : N - p;
      const c10::complex<T> wp = std::conj(cuda_utils::TwiddleFactor<T>(N, p));
      const c10::complex<T> wq = std::conj(cuda_utils::TwiddleFactor<T>(N, q));
      const c10::complex<T> zxp = (src[p] + std::conj(src[q])) * T(0.5);
      const c10::complex<T> zyp =
          complex_utils::Mul1i(src[p] - std::conj(src[q])) * wp * T(-0.5);
      const c10::complex<T> zxq = (src[q] + std::conj(src[p])) * T(0.5);
      const c10::complex<T> zyq =
          complex_utils::Mul1i(src[q] - std::conj(src[p])) * wq * T(-0.5);
      dst[p] = zxp + zyp;
      dst[q] = zxq + zyq;
      if (p == 0) {
        dst[N] = zxp - zyp;
      }
    }
    if (threadIdx.x == 0) {
      constexpr int p = N / 2;
      const c10::complex<T> w = std::conj(cuda_utils::TwiddleFactor<T>(N, p));
      const c10::complex<T> zx = c10::complex<T>(src[p].real(), T(0));
      const c10::complex<T> zy = c10::complex<T>(src[p].imag(), T(0)) * w;
      dst[p] = zx + zy;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      const int p = i * kNumThreads + threadIdx.x;
      const int q = p == 0 ? 0 : N - p;
      const c10::complex<T> w = std::conj(cuda_utils::TwiddleFactor<T>(N, p));
      const c10::complex<T> z1 = src[p];
      const c10::complex<T> z2 = std::conj(src[q]);
      const c10::complex<T> zx = (z1 + z2) * T(0.5);
      const c10::complex<T> zy = complex_utils::Mul1i(z1 - z2) * w * T(-0.5);
      dst[p] = zx + zy;
      if (p == 0) {
        dst[N] = zx - zy;
      }
    }
  }
}

template <typename T, int N, int kNumThreads>
__inline__ __device__ void IRFFTPreProcess(const c10::complex<T>* src,
                                           c10::complex<T>* dst) {
  constexpr int kElementsPerThread = N / kNumThreads;
  constexpr int kNumBits = Log2(N);

  if (src == dst) {
    assert(kElementsPerThread > 1);
#pragma unroll
    for (int i = 0; i < kElementsPerThread / 2; ++i) {
      const int p = i * kNumThreads + threadIdx.x;
      const int q = N - p;
      const c10::complex<T> wp = cuda_utils::TwiddleFactor<T>(N, p);
      const c10::complex<T> wq = cuda_utils::TwiddleFactor<T>(N, q);
      const c10::complex<T> zxp = (src[p] + std::conj(src[q]));
      const c10::complex<T> zyp = (src[p] - std::conj(src[q])) * wp;
      const c10::complex<T> zxq = (src[q] + std::conj(src[p]));
      const c10::complex<T> zyq = (src[q] - std::conj(src[p])) * wq;
      dst[p] = (zxp + complex_utils::Mul1i(zyp)) * T(0.5);
      dst[q] = (zxq + complex_utils::Mul1i(zyq)) * T(0.5);
    }
    if (threadIdx.x == 0) {
      constexpr int p = N / 2;
      const c10::complex<T> w = cuda_utils::TwiddleFactor<T>(N, p);
      const c10::complex<T> zx = c10::complex<T>(src[p].real(), T(0));
      const c10::complex<T> zy = c10::complex<T>(T(0), src[p].imag()) * w;
      dst[p] = (zx + complex_utils::Mul1i(zy));
    }
    __syncthreads();
    BitReverse<T, N, kNumThreads>(dst, dst);
  } else {
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      const int idx = i * kNumThreads + threadIdx.x;
      const int rev = (__brev(idx) >> (32 - kNumBits));
      const c10::complex<T> w = cuda_utils::TwiddleFactor<T>(N, idx);
      const c10::complex<T> z1 = src[idx];
      const c10::complex<T> z2 = std::conj(src[N - idx]);
      const c10::complex<T> zx = (z1 + z2);
      const c10::complex<T> zy = (z1 - z2) * w;
      dst[rev] = (zx + complex_utils::Mul1i(zy)) * T(0.5);
    }
  }
}

// https://kovleventer.com/blog/fft_real/
template <typename T, typename T_ACC, int N, int kNumThreads>
__inline__ __device__ void BlockRFFT(const T* x, int size, bool flip,
                                     c10::complex<T_ACC>* y,
                                     c10::complex<T_ACC>* shm) {
  if (flip) {
    LoadAsComplex<T, T_ACC, N, kNumThreads, true>(x, size, shm);
  } else {
    LoadAsComplex<T, T_ACC, N, kNumThreads, false>(x, size, shm);
  }
  __syncthreads();

  BlockFFTImpl<T_ACC, N, /*kIFFT=*/false>(shm);
  __syncthreads();

  RFFTPostProcess<T_ACC, N, kNumThreads>(shm, y);
}

template <typename T, typename T_ACC, int N, int kNumThreads>
__inline__ __device__ void BlockIRFFT(const c10::complex<T_ACC>* x, int size,
                                      bool flip, T* y,
                                      c10::complex<T_ACC>* shm) {
  IRFFTPreProcess<T_ACC, N, kNumThreads>(x, shm);
  __syncthreads();

  BlockFFTImpl<T_ACC, N, /*kIFFT=*/true>(shm);
  __syncthreads();

  constexpr T_ACC coef = T_ACC(1) / static_cast<T_ACC>(N);
  if (flip) {
    SaveAsReal<T, T_ACC, N, kNumThreads, true>(shm, size, coef, y);
  } else {
    SaveAsReal<T, T_ACC, N, kNumThreads, false>(shm, size, coef, y);
  }
}

}  // namespace fft
}  // namespace megalodon
