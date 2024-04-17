#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/MemoryFormat.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "blas.h"
#include "cuda_utils.cuh"
#include "ops/attention.h"
#include "random_utils.cuh"
#include "softmax.cuh"

namespace megalodon {
namespace ops {

namespace {

template <typename T>
__global__ void MakeGemmBatchedInputsCUDAKernel(
    int64_t B, int64_t N, const T* a, int64_t outer_stride_a,
    int64_t inner_stride_a, const T* b, int64_t outer_stride_b,
    int64_t inner_stride_b, T* c, int64_t outer_stride_c,
    int64_t inner_stride_c, const T** a_ptr, const T** b_ptr, T** c_ptr) {
  for (int64_t i = threadIdx.x; i < B * N; i += blockDim.x) {
    const int64_t x = i / N;
    const int64_t y = i % N;
    a_ptr[i] = a + x * outer_stride_a + y * inner_stride_a;
    b_ptr[i] = b + x * outer_stride_b + y * inner_stride_b;
    c_ptr[i] = c + x * outer_stride_c + y * inner_stride_c;
  }
}

template <typename T>
void AttentionCUDAFwdImpl(const torch::Tensor& q, const torch::Tensor& k,
                          const torch::Tensor& v, double scale, double dropout,
                          bool use_causal_mask, torch::Tensor& y,
                          torch::Tensor& w) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t B = q.size(0);
  const int64_t L1 = q.size(1);
  const int64_t L2 = k.size(1);
  const int64_t N = q.size(2);
  const int64_t H1 = q.size(3);
  const int64_t H2 = v.size(3);

  const T* q_data = q.data_ptr<T>();
  const T* k_data = k.data_ptr<T>();
  const T* v_data = v.data_ptr<T>();
  T* y_data = y.data_ptr<T>();
  T* w_data = w.data_ptr<T>();

  const int64_t ptr_size = B * N * sizeof(T*);
  torch::Tensor a_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  torch::Tensor b_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  torch::Tensor c_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  const T** a_ptr_data = reinterpret_cast<const T**>(a_ptr.data_ptr());
  const T** b_ptr_data = reinterpret_cast<const T**>(b_ptr.data_ptr());
  T** c_ptr_data = reinterpret_cast<T**>(c_ptr.data_ptr());

  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kT, B, L1, L2, H1,
        static_cast<T_ACC>(scale), q_data, H1, L1 * H1, k_data, H1, L2 * H1,
        /*beta=*/T_ACC(0), w_data, L2, L1 * L2);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, q_data, L1 * N * H1, H1, k_data, L2 * N * H1, H1, w_data,
            N * L1 * L2, L1 * L2, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kT, B * N, L1, L2, H1,
        static_cast<T_ACC>(scale), a_ptr_data, N * H1, b_ptr_data, N * H1,
        /*beta=*/T_ACC(0), c_ptr_data, L2);
  }

  constexpr int64_t kShmSize = 0;
  const int64_t batch_size = B * N;
  const int64_t outer_size = L1;
  const int64_t inner_size = L2;
  if (dropout == 0.0) {
    DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
        softmax::AttentionSoftmaxFwdKernel, T, T_ACC,
        dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
        outer_size, inner_size, use_causal_mask, w_data, w_data);
  } else {
    const int64_t random_capacity =
        std::max(int64_t(1) << utils::CeilLog2(inner_size),
                 cuda_utils::kWarpSize * random_utils::kRandomUnroll);
    auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(random_capacity);
    }
    DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
        softmax::AttentionDropKeySoftmaxFwdKernel, T, T_ACC,
        dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
        rng_engine_inputs, outer_size, inner_size, static_cast<T_ACC>(dropout),
        use_causal_mask, w_data, w_data);
  }

  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kN, B, L1, H2, L2,
        /*alpha=*/T_ACC(1), w_data, L2, L1 * L2, v_data, H2, L2 * H2,
        /*beta=*/T_ACC(0), y_data, H2, L1 * H2);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, w_data, N * L1 * L2, L1 * L2, v_data, L2 * N * H2, H2, y_data,
            L1 * N * H2, H2, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(handle, blas::TransposeOp::kN,
                             blas::TransposeOp::kN, B * N, L1, H2, L2,
                             /*alpha=*/T_ACC(1), a_ptr_data, L2, b_ptr_data,
                             N * H2, /*beta=*/T_ACC(0), c_ptr_data, N * H2);
  }
}

template <typename T>
void AttentionCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& q,
                          const torch::Tensor& k, const torch::Tensor& v,
                          const torch::Tensor& w, double scale,
                          bool use_causal_mask, torch::Tensor& q_grad,
                          torch::Tensor& k_grad, torch::Tensor& v_grad) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t B = q.size(0);
  const int64_t L1 = q.size(1);
  const int64_t L2 = k.size(1);
  const int64_t N = q.size(2);
  const int64_t H1 = q.size(3);
  const int64_t H2 = v.size(3);

  torch::Tensor w_grad = torch::empty_like(w);

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* q_data = q.data_ptr<T>();
  const T* k_data = k.data_ptr<T>();
  const T* v_data = v.data_ptr<T>();
  const T* w_data = w.data_ptr<T>();
  T* q_grad_data = q_grad.data_ptr<T>();
  T* k_grad_data = k_grad.data_ptr<T>();
  T* v_grad_data = v_grad.data_ptr<T>();
  T* w_grad_data = w_grad.data_ptr<T>();

  const int64_t ptr_size = B * N * sizeof(T*);
  torch::Tensor a_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  torch::Tensor b_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  torch::Tensor c_ptr = torch::empty({ptr_size}, q.options().dtype(at::kByte));
  const T** a_ptr_data = reinterpret_cast<const T**>(a_ptr.data_ptr());
  const T** b_ptr_data = reinterpret_cast<const T**>(b_ptr.data_ptr());
  T** c_ptr_data = reinterpret_cast<T**>(c_ptr.data_ptr());

  at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // dL/dv
  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kT, blas::TransposeOp::kN, B, L2, H2, L1,
        /*alpha=*/T_ACC(1), w_data, L2, L1 * L2, y_grad_data, H2, L1 * H2,
        /*beta=*/T_ACC(0), v_grad_data, H2, L2 * H2);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, w_data, N * L1 * L2, L1 * L2, y_grad_data, L1 * N * H2, H2,
            v_grad_data, L2 * N * H2, H2, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(handle, blas::TransposeOp::kT,
                             blas::TransposeOp::kN, B * N, L2, H2, L1,
                             /*alpha=*/T_ACC(1), a_ptr_data, L2, b_ptr_data,
                             N * H2, /*beta=*/T_ACC(0), c_ptr_data, N * H2);
  }

  // dL/dw
  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kT, B, L1, L2, H2,
        /*alpha=*/T_ACC(1), y_grad_data, H2, L1 * H2, v_data, H2, L2 * H2,
        /*beta=*/T_ACC(0), w_grad_data, L2, L1 * L2);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, y_grad_data, L1 * N * H2, H2, v_data, L2 * N * H2, H2,
            w_grad_data, N * L1 * L2, L1 * L2, a_ptr_data, b_ptr_data,
            c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(handle, blas::TransposeOp::kN,
                             blas::TransposeOp::kT, B * N, L1, L2, H2,
                             /*alpha=*/T_ACC(1), a_ptr_data, N * H2, b_ptr_data,
                             N * H2, /*beta=*/T_ACC(0), c_ptr_data, L2);
  }

  constexpr int64_t kShmSize = 0;
  const int64_t batch_size = B * N;
  const int64_t outer_size = L1;
  const int64_t inner_size = L2;
  DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
      softmax::AttentionSoftmaxBwdKernel, T, T_ACC,
      dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
      outer_size, inner_size, use_causal_mask, w_grad_data, w_data,
      w_grad_data);

  // dL/dk
  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kT, blas::TransposeOp::kN, B, L2, H1, L1,
        static_cast<T_ACC>(scale), w_grad_data, L2, L1 * L2, q_data, H1,
        L1 * H1, /*beta=*/T_ACC(0), k_grad_data, H1, L2 * H1);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, w_grad_data, N * L1 * L2, L1 * L2, q_data, L1 * N * H1, H1,
            k_grad_data, L2 * N * H1, H1, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(
        handle, blas::TransposeOp::kT, blas::TransposeOp::kN, B * N, L2, H1, L1,
        static_cast<T_ACC>(scale), a_ptr_data, L2, b_ptr_data, N * H1,
        /*beta=*/T_ACC(0), c_ptr_data, N * H1);
  }

  // dL/dq
  if (N == 1) {
    blas::GemmStridedBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kN, B, L1, H1, L2,
        static_cast<T_ACC>(scale), w_grad_data, L2, L1 * L2, k_data, H1,
        L2 * H1, /*beta=*/T_ACC(0), q_grad_data, H1, L1 * H1);
  } else {
    MakeGemmBatchedInputsCUDAKernel<T>
        <<<1, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, N, w_grad_data, N * L1 * L2, L1 * L2, k_data, L2 * N * H1, H1,
            q_grad_data, L1 * N * H1, H1, a_ptr_data, b_ptr_data, c_ptr_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blas::GemmBatchedCUDA<T>(
        handle, blas::TransposeOp::kN, blas::TransposeOp::kN, B * N, L1, H1, L2,
        static_cast<T_ACC>(scale), a_ptr_data, L2, b_ptr_data, N * H1,
        /*beta=*/T_ACC(0), c_ptr_data, N * H1);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> AttentionCUDAFwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, double dropout, bool use_causal_mask) {
  const int64_t B = q.size(0);
  const int64_t L1 = q.size(1);
  const int64_t L2 = k.size(1);
  const int64_t N = q.size(2);
  const int64_t H2 = v.size(3);
  torch::Tensor y = torch::empty(
      {B, L1, N, H2}, v.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor w = torch::empty(
      {B, N, L1, L2}, v.options().memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "AttentionCUDAFwd", [&]() {
        AttentionCUDAFwdImpl<scalar_t>(
            *(q.expect_contiguous()), *(k.expect_contiguous()),
            *(v.expect_contiguous()), scale, dropout, use_causal_mask, y, w);
      });
  return std::make_tuple<torch::Tensor, torch::Tensor>(std::move(y),
                                                       std::move(w));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AttentionCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& w, double scale,
    bool use_causal_mask) {
  torch::Tensor q_grad = torch::empty_like(
      q, q.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor k_grad = torch::empty_like(
      k, k.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor v_grad = torch::empty_like(
      v, v.options().memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "AttentionCUDABwd", [&]() {
        AttentionCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(q.expect_contiguous()),
            *(k.expect_contiguous()), *(v.expect_contiguous()),
            *(w.expect_contiguous()), scale, use_causal_mask, q_grad, k_grad,
            v_grad);
      });
  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
      std::move(q_grad), std::move(k_grad), std::move(v_grad));
}

}  // namespace ops
}  // namespace megalodon
