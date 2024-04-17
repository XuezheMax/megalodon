#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <type_traits>
#include <vector>

#include "cuda_utils.cuh"
#include "fft.cuh"
#include "ops/fftconv.h"
#include "utils.h"

namespace megalodon {
namespace ops {

namespace {

template <typename T, typename T_ACC, int N, int kNumThreads>
__global__ void RFFTCUDAKernel(const T* __restrict__ x, int size, bool flip,
                               c10::complex<T_ACC>* __restrict__ y) {
  extern __shared__ float shm[];
  c10::complex<T_ACC>* shm_ptr = reinterpret_cast<c10::complex<T_ACC>*>(shm);

  const int b = blockIdx.x;
  fft::BlockRFFT<T, T_ACC, N, kNumThreads>(x + b * size, N, flip,
                                           y + b * (N + 1), shm_ptr);
}

template <typename T, typename T_ACC, int N, int kNumThreads>
__global__ void FFTConvCUDAFwdKernel(
    int H, int L, const T* __restrict__ x,
    const c10::complex<T_ACC>* __restrict__ k_f, T* __restrict__ y,
    c10::complex<T_ACC>* __restrict__ x_f) {
  constexpr int kElementsPerThread = N / kNumThreads;
  extern __shared__ float shm[];
  c10::complex<T_ACC>* shm_ptr = reinterpret_cast<c10::complex<T_ACC>*>(shm);

  const int b = blockIdx.y;
  const int h = blockIdx.x;
  const T* x_ptr = x + (b * H + h) * L;
  const c10::complex<T_ACC>* k_f_ptr = k_f + h * (N + 1);
  T* y_ptr = y + (b * H + h) * L;
  c10::complex<T_ACC>* x_f_ptr = x_f + (b * H + h) * (N + 1);

  if constexpr (N == 16384) {
    fft::BlockRFFT<T, T_ACC, N, kNumThreads>(x_ptr, L, /*flip=*/false, shm_ptr,
                                             shm_ptr);
  } else {
    fft::BlockRFFT<T, T_ACC, N, kNumThreads>(x_ptr, L, /*flip=*/false, shm_ptr,
                                             shm_ptr + N);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * kNumThreads + threadIdx.x;
    x_f_ptr[idx] = shm_ptr[idx];
    shm_ptr[idx] *= k_f_ptr[idx];
  }
  if (threadIdx.x == 0) {
    x_f_ptr[N] = shm_ptr[N];
    shm_ptr[N] *= k_f_ptr[N];
  }
  __syncthreads();

  if constexpr (N == 16384) {
    fft::BlockIRFFT<T, T_ACC, N, kNumThreads>(shm_ptr, L, /*flip=*/false, y_ptr,
                                              shm_ptr);
  } else {
    fft::BlockIRFFT<T, T_ACC, N, kNumThreads>(shm_ptr, L, /*flip=*/false, y_ptr,
                                              shm_ptr + N);
  }
}

template <typename T, typename T_ACC, int N, int kNumThreads>
__global__ void FFTConvCUDABwdKernel(
    int H, int L, const T* __restrict__ y_grad,
    const c10::complex<T_ACC>* __restrict__ x_f,
    const c10::complex<T_ACC>* __restrict__ k_f, T* __restrict__ x_grad,
    c10::complex<T_ACC>* __restrict__ k_grad_f) {
  constexpr int kElementsPerThread = N / kNumThreads;
  extern __shared__ float shm[];
  c10::complex<T_ACC>* shm_ptr = reinterpret_cast<c10::complex<T_ACC>*>(shm);

  const int b = blockIdx.y;
  const int h = blockIdx.x;
  const T* y_grad_ptr = y_grad + (b * H + h) * L;
  const c10::complex<T_ACC>* x_f_ptr = x_f + (b * H + h) * (N + 1);
  const c10::complex<T_ACC>* k_f_ptr = k_f + h * (N + 1);
  T* x_grad_ptr = x_grad + (b * H + h) * L;
  c10::complex<T_ACC>* k_grad_f_ptr = k_grad_f + (b * H + h) * (N + 1);

  if constexpr (N == 16384) {
    fft::BlockRFFT<T, T_ACC, N, kNumThreads>(y_grad_ptr, L, /*flip=*/true,
                                             shm_ptr, shm_ptr);
  } else {
    fft::BlockRFFT<T, T_ACC, N, kNumThreads>(y_grad_ptr, L, /*flip=*/true,
                                             shm_ptr, shm_ptr + N);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * kNumThreads + threadIdx.x;
    k_grad_f_ptr[idx] = shm_ptr[idx] * x_f_ptr[idx];
    shm_ptr[idx] *= k_f_ptr[idx];
  }
  if (threadIdx.x == 0) {
    k_grad_f_ptr[N] = shm_ptr[N] * x_f_ptr[N];
    shm_ptr[N] *= k_f_ptr[N];
  }
  __syncthreads();

  if constexpr (N == 16384) {
    fft::BlockIRFFT<T, T_ACC, N, kNumThreads>(shm_ptr, L, /*flip=*/true,
                                              x_grad_ptr, shm_ptr);
  } else {
    fft::BlockIRFFT<T, T_ACC, N, kNumThreads>(shm_ptr, L, /*flip=*/true,
                                              x_grad_ptr, shm_ptr + N);
  }
}

template <typename T, typename T_ACC, int N, int kNumThreads>
__global__ void FFTConvKernelCUDABwdKernel(
    int B, int H, int L, const c10::complex<T_ACC>* __restrict__ k_grad_f,
    T* __restrict__ k_grad) {
  constexpr int kElementsPerThread = N / kNumThreads;
  extern __shared__ float shm[];
  c10::complex<T_ACC>* shm_ptr = reinterpret_cast<c10::complex<T_ACC>*>(shm);

  const int h = blockIdx.x;
  T* k_grad_ptr = k_grad + h * L;

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * kNumThreads + threadIdx.x;
    c10::complex<T_ACC> sum(T_ACC(0));
    for (int b = 0; b < B; ++b) {
      sum += k_grad_f[(b * H + h) * (N + 1) + idx];
    }
    shm_ptr[idx] = sum;
  }
  if (threadIdx.x == 0) {
    c10::complex<T_ACC> sum(T_ACC(0));
    for (int b = 0; b < B; ++b) {
      sum += k_grad_f[(b * H + h) * (N + 1) + N];
    }
    shm_ptr[N] = sum;
  }
  __syncthreads();

  if (N == 16384) {
    fft::BlockIRFFT<T, T_ACC, N, kNumThreads>(shm_ptr, L, /*flip=*/true,
                                              k_grad_ptr, shm_ptr);
  } else {
    fft::BlockIRFFT<T, T_ACC, N, kNumThreads>(shm_ptr, L, /*flip=*/true,
                                              k_grad_ptr, shm_ptr + N);
  }
}

int64_t ComputeFFTSize(int64_t N) {
  return std::max((int64_t(1) << utils::CeilLog2(N)), cuda_utils::kWarpSize);
}

template <typename T>
void RFFTCUDAImpl(const torch::Tensor& x, bool flip, torch::Tensor& y) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t N = x.size(-1);
  const int64_t B = x.numel() / N;
  const int64_t fft_size = ComputeFFTSize(N);

  const T* x_data = x.data_ptr<T>();
  c10::complex<T_ACC>* y_data = y.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t shm_size = fft_size * sizeof(c10::complex<T_ACC>);

  if (fft_size == 32) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 32, 32>, B, 32, shm_size,
                             cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 64) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 64, 32>, B, 32, shm_size,
                             cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 128) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 128, 64>, B, 64, shm_size,
                             cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 256) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 256, 128>, B, 128,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 512) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 512, 256>, B, 256,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 1024) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 1024, 512>, B, 512,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 2048) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 2048, 1024>, B, 1024,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 4096) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 4096, 1024>, B, 1024,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 8192) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 8192, 1024>, B, 1024,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else if (fft_size == 16384) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 16384, 1024>, B, 1024,
                             shm_size, cuda_stream, x_data, N, flip, y_data);
  } else {
    TORCH_CHECK(false);
  }
}

template <typename T>
void FFTConvCUDAFwdImpl(const torch::Tensor& x, const torch::Tensor& k_f,
                        torch::Tensor& y, torch::Tensor& x_f) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t H = x.size(1);
  const int64_t L = x.size(2);
  const int64_t fft_size = ComputeFFTSize(L);

  const T* x_data = x.data_ptr<T>();
  const c10::complex<T_ACC>* k_f_data = k_f.data_ptr<c10::complex<T_ACC>>();
  T* y_data = y.data_ptr<T>();
  c10::complex<T_ACC>* x_f_data = x_f.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t shm_size =
      (fft_size == 16384 ? (fft_size + 1) : (fft_size * 2)) *
      sizeof(c10::complex<T_ACC>);

  if (fft_size == 32) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 32, 32>, dim3(H, B),
                             32, shm_size, cuda_stream, H, L, x_data, k_f_data,
                             y_data, x_f_data);
  } else if (fft_size == 64) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 64, 32>, dim3(H, B),
                             32, shm_size, cuda_stream, H, L, x_data, k_f_data,
                             y_data, x_f_data);
  } else if (fft_size == 128) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 128, 64>,
                             dim3(H, B), 64, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 256) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 256, 128>,
                             dim3(H, B), 128, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 512) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 512, 256>,
                             dim3(H, B), 256, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 1024) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 1024, 512>,
                             dim3(H, B), 512, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 2048) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 2048, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 4096) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 4096, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 8192) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 8192, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else if (fft_size == 16384) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 16384, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             x_data, k_f_data, y_data, x_f_data);
  } else {
    TORCH_CHECK(false);
  }
}

template <typename T>
void FFTConvCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& x_f,
                        const torch::Tensor& k_f, const torch::Dtype& k_dtype,
                        torch::Tensor& x_grad, torch::Tensor& k_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = y_grad.size(0);
  const int64_t H = y_grad.size(1);
  const int64_t L = y_grad.size(2);
  const int64_t fft_size = x_f.size(2) - 1;

  torch::Tensor k_grad_f = torch::empty_like(x_f);

  const T* y_grad_data = y_grad.data_ptr<T>();
  const c10::complex<T_ACC>* x_f_data = x_f.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* k_f_data = k_f.data_ptr<c10::complex<T_ACC>>();
  T* x_grad_data = x_grad.data_ptr<T>();
  c10::complex<T_ACC>* k_grad_f_data = k_grad_f.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x_f));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t shm_size =
      (fft_size == 16384 ? (fft_size + 1) : (fft_size * 2)) *
      sizeof(c10::complex<T_ACC>);

  if (fft_size == 32) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 32, 32>, dim3(H, B),
                             32, shm_size, cuda_stream, H, L, y_grad_data,
                             x_f_data, k_f_data, x_grad_data, k_grad_f_data);

    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 32, 32>, H,
                               32, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 32, 32>,
                               H, 32, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 64) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 64, 32>, dim3(H, B),
                             32, shm_size, cuda_stream, H, L, y_grad_data,
                             x_f_data, k_f_data, x_grad_data, k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 64, 32>, H,
                               32, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 64, 32>,
                               H, 32, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 128) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 128, 64>,
                             dim3(H, B), 64, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 128, 64>, H,
                               64, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());

    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 128, 64>, H, 64, shm_size,
          cuda_stream, B, H, L, k_grad_f_data, k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 256) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 256, 128>,
                             dim3(H, B), 128, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 256, 128>,
                               H, 128, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 256, 128>, H, 128, shm_size,
          cuda_stream, B, H, L, k_grad_f_data, k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 512) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 512, 256>,
                             dim3(H, B), 256, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 512, 256>,
                               H, 256, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 512, 256>, H, 256, shm_size,
          cuda_stream, B, H, L, k_grad_f_data, k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 1024) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 1024, 512>,
                             dim3(H, B), 512, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 1024, 512>,
                               H, 512, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 1024, 512>, H, 512, shm_size,
          cuda_stream, B, H, L, k_grad_f_data, k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 2048) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 2048, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 2048, 1024>,
                               H, 1024, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 2048, 1024>, H, 1024,
          shm_size, cuda_stream, B, H, L, k_grad_f_data,
          k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 4096) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 4096, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 4096, 1024>,
                               H, 1024, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 4096, 1024>, H, 1024,
          shm_size, cuda_stream, B, H, L, k_grad_f_data,
          k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 8192) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 8192, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 8192, 1024>,
                               H, 1024, shm_size, cuda_stream, B, H, L,
                               k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 8192, 1024>, H, 1024,
          shm_size, cuda_stream, B, H, L, k_grad_f_data,
          k_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 16384) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 16384, 1024>,
                             dim3(H, B), 1024, shm_size, cuda_stream, H, L,
                             y_grad_data, x_f_data, k_f_data, x_grad_data,
                             k_grad_f_data);
    if (k_dtype == y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T, T_ACC, 16384, 1024>, H, 1024, shm_size,
          cuda_stream, B, H, L, k_grad_f_data, k_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 16384, 1024>, H, 1024,
          shm_size, cuda_stream, B, H, L, k_grad_f_data,
          k_grad.data_ptr<T_ACC>());
    }
  } else {
    TORCH_CHECK(false);
  }
}

}  // namespace

torch::Tensor RFFTCUDA(const torch::Tensor& x, bool flip) {
  std::vector<int64_t> sizes = x.sizes().vec();
  const int64_t L = sizes.back();
  const int64_t rfft_size = ComputeFFTSize(L) + 1;
  sizes.back() = rfft_size;
  TORCH_CHECK(L <= fft::kFFTMaxLength);
  const auto complex_type =
      x.scalar_type() == at::kDouble ? at::kComplexDouble : at::kComplexFloat;
  torch::Tensor y =
      torch::empty(sizes, x.options()
                              .dtype(complex_type)
                              .memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "RFFT",
      [&]() { RFFTCUDAImpl<scalar_t>(*(x.expect_contiguous()), flip, y); });
  return y;
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDAFwd(
    const torch::Tensor& x, const torch::Tensor& k_f) {
  std::vector<int64_t> sizes = x.sizes().vec();
  const int64_t L = sizes.back();
  const int64_t rfft_size = ComputeFFTSize(L) + 1;
  sizes.back() = rfft_size;
  TORCH_CHECK(L <= fft::kFFTMaxLength);
  const auto complex_type =
      x.scalar_type() == at::kDouble ? at::kComplexDouble : at::kComplexFloat;
  torch::Tensor y = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor x_f =
      torch::empty(sizes, x.options()
                              .dtype(complex_type)
                              .memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "FFTConvFwd", [&]() {
        FFTConvCUDAFwdImpl<scalar_t>(*(x.expect_contiguous()),
                                     *(k_f.expect_contiguous()), y, x_f);
      });
  return std::make_tuple(y, x_f);
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x_f,
    const torch::Tensor& k_f, const torch::Dtype& k_dtype) {
  const int64_t H = y_grad.size(1);
  const int64_t L = y_grad.size(2);
  torch::Tensor x_grad = torch::empty_like(
      y_grad, y_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor k_grad =
      torch::empty({H, L}, y_grad.options().dtype(k_dtype).memory_format(
                               at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, y_grad.scalar_type(), "FFTConvBwd", [&]() {
        FFTConvCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x_f.expect_contiguous()),
            *(k_f.expect_contiguous()), k_dtype, x_grad, k_grad);
      });
  return std::make_tuple(x_grad, k_grad);
}

}  // namespace ops
}  // namespace megalodon
