#pragma once

#include <c10/core/ScalarType.h>

#include <cstdint>
#include <type_traits>

#include "cuda_utils.cuh"

namespace megalodon {
namespace register_utils {

template <typename SRC, typename DST, int64_t kElementsPerThread>
__inline__ __device__ void LoadImpl(const SRC* __restrict__ src, int64_t size,
                                    DST default_value, DST* __restrict__ dst) {
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    dst[i] = idx < size ? static_cast<DST>(src[idx]) : default_value;
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void LoadFP32x2Impl(const float* __restrict__ src,
                                          int64_t size, float default_value,
                                          float* __restrict__ dst) {
  const float2 default2 = {default_value, default_value};
  const float2* src2 = reinterpret_cast<const float2*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = idx * 2 < size ? src2[idx] : default2;
    dst[i * 2 + 0] = idx * 2 + 0 < size ? v2.x : default_value;
    dst[i * 2 + 1] = idx * 2 + 1 < size ? v2.y : default_value;
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void LoadFP32x4Impl(const float* __restrict__ src,
                                          int64_t size, float default_value,
                                          float* __restrict__ dst) {
  const float4 default4 = {default_value, default_value, default_value,
                           default_value};
  const float4* src4 = reinterpret_cast<const float4*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 4; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float4 v4 = idx * 4 < size ? src4[idx] : default4;
    dst[i * 4 + 0] = idx * 4 + 0 < size ? v4.x : default_value;
    dst[i * 4 + 1] = idx * 4 + 1 < size ? v4.y : default_value;
    dst[i * 4 + 2] = idx * 4 + 2 < size ? v4.z : default_value;
    dst[i * 4 + 3] = idx * 4 + 3 < size ? v4.w : default_value;
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void LoadFP16x2Impl(const at::Half* __restrict__ src,
                                          int64_t size, float default_value,
                                          float* __restrict__ dst) {
  const float2 default2 = {default_value, default_value};
  const __half2* src2 = reinterpret_cast<const __half2*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = idx * 2 < size ? __half22float2(src2[idx]) : default2;
    dst[i * 2 + 0] = idx * 2 + 0 < size ? v2.x : default_value;
    dst[i * 2 + 1] = idx * 2 + 1 < size ? v2.y : default_value;
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void LoadBF16x2Impl(const at::BFloat16* __restrict__ src,
                                          int64_t size, float default_value,
                                          float* __restrict__ dst) {
  const float2 default2 = {default_value, default_value};
  const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const float2 v2 = idx * 2 < size ? __bfloat1622float2(src2[idx]) : default2;
#else
    const __nv_bfloat162 x2 = src2[idx];
    const float2 v2 = idx * 2 < size ? make_float2(__bfloat162float(x2.x),
                                                   __bfloat162float(x2.y))
                                     : default2;
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst[i * 2 + 0] = idx * 2 + 0 < size ? v2.x : default_value;
    dst[i * 2 + 1] = idx * 2 + 1 < size ? v2.y : default_value;
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void LoadBF16x4Impl(const at::BFloat16* __restrict__ src,
                                          int64_t size, float default_value,
                                          float* __restrict__ dst) {
  const __nv_bfloat16 default_bf16 = __float2bfloat16(default_value);
  const cuda_utils::BF16x4 default4 = {default_bf16, default_bf16, default_bf16,
                                       default_bf16};
  const cuda_utils::BF16x4* src4 =
      reinterpret_cast<const cuda_utils::BF16x4*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 4; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const cuda_utils::BF16x4 v4 = idx * 4 < size ? src4[idx] : default4;
    dst[i * 4 + 0] =
        idx * 4 + 0 < size ? __bfloat162float(v4.x0) : default_value;
    dst[i * 4 + 1] =
        idx * 4 + 1 < size ? __bfloat162float(v4.x1) : default_value;
    dst[i * 4 + 2] =
        idx * 4 + 2 < size ? __bfloat162float(v4.x2) : default_value;
    dst[i * 4 + 3] =
        idx * 4 + 3 < size ? __bfloat162float(v4.x3) : default_value;
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void LoadBF16x8Impl(const at::BFloat16* __restrict__ src,
                                          int64_t size, float default_value,
                                          float* __restrict__ dst) {
  const __nv_bfloat16 default_bf16 = __float2bfloat16(default_value);
  const cuda_utils::BF16x8 default8 = {default_bf16, default_bf16, default_bf16,
                                       default_bf16, default_bf16, default_bf16,
                                       default_bf16, default_bf16};
  const cuda_utils::BF16x8* src8 =
      reinterpret_cast<const cuda_utils::BF16x8*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 8; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const cuda_utils::BF16x8 v8 = idx * 8 < size ? src8[idx] : default8;
    dst[i * 8 + 0] =
        idx * 8 + 0 < size ? __bfloat162float(v8.x0) : default_value;
    dst[i * 8 + 1] =
        idx * 8 + 1 < size ? __bfloat162float(v8.x1) : default_value;
    dst[i * 8 + 2] =
        idx * 8 + 2 < size ? __bfloat162float(v8.x2) : default_value;
    dst[i * 8 + 3] =
        idx * 8 + 3 < size ? __bfloat162float(v8.x3) : default_value;
    dst[i * 8 + 4] =
        idx * 8 + 4 < size ? __bfloat162float(v8.x4) : default_value;
    dst[i * 8 + 5] =
        idx * 8 + 5 < size ? __bfloat162float(v8.x5) : default_value;
    dst[i * 8 + 6] =
        idx * 8 + 6 < size ? __bfloat162float(v8.x6) : default_value;
    dst[i * 8 + 7] =
        idx * 8 + 7 < size ? __bfloat162float(v8.x7) : default_value;
  }
}

template <typename SRC, typename DST, int64_t kElementsPerThread>
__inline__ __device__ void SaveImpl(const SRC* __restrict__ src, int64_t size,
                                    DST* __restrict__ dst) {
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx < size) {
      dst[idx] = static_cast<DST>(src[i]);
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void SaveFP32x2Impl(const float* __restrict__ src,
                                          int64_t size,
                                          float* __restrict__ dst) {
  float2* dst2 = reinterpret_cast<float2*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = {src[i * 2 + 0], src[i * 2 + 1]};
    if (idx * 2 < size) {
      dst2[idx] = v2;
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void SaveFP32x4Impl(const float* __restrict__ src,
                                          int64_t size,
                                          float* __restrict__ dst) {
  float4* dst4 = reinterpret_cast<float4*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 4; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float4 v4 = {src[i * 4 + 0], src[i * 4 + 1], src[i * 4 + 2],
                       src[i * 4 + 3]};
    if (idx * 4 < size) {
      dst4[idx] = v4;
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void SaveFP16x2Impl(const float* __restrict__ src,
                                          int64_t size,
                                          at::Half* __restrict__ dst) {
  __half2* dst2 = reinterpret_cast<__half2*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = {src[i * 2 + 0], src[i * 2 + 1]};
    if (idx * 2 < size) {
      dst2[idx] = __float22half2_rn(v2);
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void SaveBF16x2Impl(const float* __restrict__ src,
                                          int64_t size,
                                          at::BFloat16* __restrict__ dst) {
  __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = {src[i * 2 + 0], src[i * 2 + 1]};
    if (idx * 2 < size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      dst2[idx] = __float22bfloat162_rn(v2);
#else
      dst2[idx] =
          make_bfloat162(__float2bfloat16(v2.x), __float2bfloat16(v2.y));
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void SaveBF16x4Impl(const float* __restrict__ src,
                                          int64_t size,
                                          at::BFloat16* __restrict__ dst) {
  cuda_utils::BF16x4* dst4 = reinterpret_cast<cuda_utils::BF16x4*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 4; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx * 4 < size) {
      dst4[idx] = {
          __float2bfloat16(src[i * 4 + 0]), __float2bfloat16(src[i * 4 + 1]),
          __float2bfloat16(src[i * 4 + 2]), __float2bfloat16(src[i * 4 + 3])};
    }
  }
}

template <int64_t kElementsPerThread>
__inline__ __device__ void SaveBF16x8Impl(const float* __restrict__ src,
                                          int64_t size,
                                          at::BFloat16* __restrict__ dst) {
  cuda_utils::BF16x8* dst8 = reinterpret_cast<cuda_utils::BF16x8*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 8; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx * 8 < size) {
      dst8[idx] = {
          __float2bfloat16(src[i * 8 + 0]), __float2bfloat16(src[i * 8 + 1]),
          __float2bfloat16(src[i * 8 + 2]), __float2bfloat16(src[i * 8 + 3]),
          __float2bfloat16(src[i * 8 + 4]), __float2bfloat16(src[i * 8 + 5]),
          __float2bfloat16(src[i * 8 + 6]), __float2bfloat16(src[i * 8 + 7])};
    }
  }
}

template <typename SRC, typename DST, int64_t kElementsPerThread>
__inline__ __device__ void Load(const SRC* __restrict__ src, int64_t size,
                                int64_t stride, DST default_value,
                                DST* __restrict__ dst) {
  if constexpr (std::is_same<SRC, float>::value &&
                std::is_same<DST, float>::value) {
    if (kElementsPerThread % 4 == 0 &&
        reinterpret_cast<std::uintptr_t>(src) % alignof(float4) == 0 &&
        stride % 4 == 0) {
      LoadFP32x4Impl<kElementsPerThread>(src, size, default_value, dst);
    } else if (kElementsPerThread % 2 == 0 &&
               reinterpret_cast<std::uintptr_t>(src) % alignof(float2) == 0 &&
               stride % 2 == 0) {
      LoadFP32x2Impl<kElementsPerThread>(src, size, default_value, dst);
    } else {
      LoadImpl<float, float, kElementsPerThread>(src, size, default_value, dst);
    }
  } else if constexpr (std::is_same<SRC, at::Half>::value &&
                       std::is_same<DST, float>::value) {
    if (kElementsPerThread % 2 == 0 &&
        reinterpret_cast<std::uintptr_t>(src) % alignof(__half2) == 0 &&
        stride % 2 == 0) {
      LoadFP16x2Impl<kElementsPerThread>(src, size, default_value, dst);
    } else {
      LoadImpl<at::Half, float, kElementsPerThread>(src, size, default_value,
                                                    dst);
    }
  } else if constexpr (std::is_same<SRC, at::BFloat16>::value &&
                       std::is_same<DST, float>::value) {
    if (kElementsPerThread % 8 == 0 &&
        reinterpret_cast<std::uintptr_t>(src) % alignof(cuda_utils::BF16x8) ==
            0 &&
        stride % 8 == 0) {
      LoadBF16x8Impl<kElementsPerThread>(src, size, default_value, dst);
    } else if (kElementsPerThread % 4 == 0 &&
               reinterpret_cast<std::uintptr_t>(src) %
                       alignof(cuda_utils::BF16x4) ==
                   0 &&
               stride % 4 == 0) {
      LoadBF16x4Impl<kElementsPerThread>(src, size, default_value, dst);
    } else if (kElementsPerThread % 2 == 0 &&
               reinterpret_cast<std::uintptr_t>(src) %
                       alignof(__nv_bfloat162) ==
                   0 &&
               stride % 2 == 0) {
      LoadBF16x2Impl<kElementsPerThread>(src, size, default_value, dst);
    } else {
      LoadImpl<at::BFloat16, float, kElementsPerThread>(src, size,
                                                        default_value, dst);
    }
  } else {
    LoadImpl<SRC, DST, kElementsPerThread>(src, size, default_value, dst);
  }
}

template <typename SRC, typename DST, int64_t kElementsPerThread>
__inline__ __device__ void Save(const SRC* __restrict__ src, int64_t size,
                                DST* __restrict__ dst) {
  if constexpr (std::is_same<SRC, float>::value &&
                std::is_same<DST, float>::value) {
    if (kElementsPerThread % 4 == 0 &&
        reinterpret_cast<std::uintptr_t>(dst) % alignof(float4) == 0 &&
        size % 4 == 0) {
      SaveFP32x4Impl<kElementsPerThread>(src, size, dst);
    } else if (kElementsPerThread % 2 == 0 &&
               reinterpret_cast<std::uintptr_t>(dst) % alignof(float2) == 0 &&
               size % 2 == 0) {
      SaveFP32x2Impl<kElementsPerThread>(src, size, dst);
    } else {
      SaveImpl<float, float, kElementsPerThread>(src, size, dst);
    }
  } else if constexpr (std::is_same<SRC, float>::value &&
                       std::is_same<DST, at::Half>::value) {
    if (kElementsPerThread % 2 == 0 &&
        reinterpret_cast<std::uintptr_t>(dst) % alignof(__half2) == 0 &&
        size % 2 == 0) {
      SaveFP16x2Impl<kElementsPerThread>(src, size, dst);
    } else {
      SaveImpl<float, at::Half, kElementsPerThread>(src, size, dst);
    }
  } else if constexpr (std::is_same<SRC, float>::value &&
                       std::is_same<DST, at::BFloat16>::value) {
    if (kElementsPerThread % 8 == 0 &&
        reinterpret_cast<std::uintptr_t>(dst) % alignof(cuda_utils::BF16x8) ==
            0 &&
        size % 8 == 0) {
      SaveBF16x8Impl<kElementsPerThread>(src, size, dst);
    } else if (kElementsPerThread % 4 == 0 &&
               reinterpret_cast<std::uintptr_t>(dst) %
                       alignof(cuda_utils::BF16x4) ==
                   0 &&
               size % 4 == 0) {
      SaveBF16x4Impl<kElementsPerThread>(src, size, dst);
    } else if (kElementsPerThread % 2 == 0 &&
               reinterpret_cast<std::uintptr_t>(dst) %
                       alignof(__nv_bfloat162) ==
                   0 &&
               size % 2 == 0) {
      SaveBF16x2Impl<kElementsPerThread>(src, size, dst);
    } else {
      SaveImpl<float, at::BFloat16, kElementsPerThread>(src, size, dst);
    }
  } else {
    SaveImpl<SRC, DST, kElementsPerThread>(src, size, dst);
  }
}

}  // namespace register_utils
}  // namespace megalodon
