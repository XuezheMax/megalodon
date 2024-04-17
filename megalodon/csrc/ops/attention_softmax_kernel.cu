#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <mutex>
#include <tuple>

#include "cuda_utils.cuh"
#include "ops/attention_softmax.h"
#include "random_utils.cuh"
#include "softmax.cuh"
#include "utils.h"

namespace megalodon {
namespace ops {

namespace {

template <typename T>
void AttentionSoftmaxCUDAFwdImpl(const torch::Tensor& x, double dropout,
                                 bool use_causal_mask, torch::Tensor& y) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  TORCH_CHECK(x.dim() >= 2);
  const int64_t outer_size = x.size(-2);
  const int64_t inner_size = x.size(-1);
  const int64_t batch_size = x.numel() / (outer_size * inner_size);
  TORCH_CHECK(inner_size <= softmax::kMaxSoftmaxSize);
  TORCH_CHECK(dropout >= 0.0 && dropout <= 1.0);

  const T* x_data = x.data_ptr<T>();
  T* y_data = y.data_ptr<T>();

  constexpr int64_t kShmSize = 0;
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (dropout == 0.0) {
    DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
        softmax::AttentionSoftmaxFwdKernel, T, T_ACC,
        dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
        outer_size, inner_size, use_causal_mask, x_data, y_data);
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
        use_causal_mask, x_data, y_data);
  }
}

template <typename T>
void AttentionSoftmaxCUDABwdImpl(const torch::Tensor& y_grad,
                                 const torch::Tensor& y, bool use_causal_mask,
                                 torch::Tensor& x_grad) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  TORCH_CHECK(y.dim() >= 2);
  const int64_t outer_size = y.size(-2);
  const int64_t inner_size = y.size(-1);
  const int64_t batch_size = y.numel() / (outer_size * inner_size);
  TORCH_CHECK(inner_size <= softmax::kMaxSoftmaxSize);

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* y_data = y.data_ptr<T>();
  T* x_grad_data = x_grad.data_ptr<T>();

  constexpr int64_t kShmSize = 0;
  at::cuda::OptionalCUDAGuard guard(at::device_of(y));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(
      softmax::AttentionSoftmaxBwdKernel, T, T_ACC,
      dim3(outer_size, batch_size), inner_size, kShmSize, cuda_stream,
      outer_size, inner_size, use_causal_mask, y_grad_data, y_data,
      x_grad_data);
}

}  // namespace

torch::Tensor AttentionSoftmaxCUDAFwd(const torch::Tensor& x, double dropout,
                                      bool use_causal_mask) {
  torch::Tensor y = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(),
                                  "AttentionSoftmaxCUDAFwd", [&]() {
                                    AttentionSoftmaxCUDAFwdImpl<scalar_t>(
                                        *(x.expect_contiguous()), dropout,
                                        use_causal_mask, y);
                                  });

  return y;
}

torch::Tensor AttentionSoftmaxCUDABwd(const torch::Tensor& y_grad,
                                      const torch::Tensor& y,
                                      bool use_causal_mask) {
  torch::Tensor x_grad = torch::empty_like(
      y, y.options().memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, y.scalar_type(),
                                  "AttentionSoftmaxCUDABwd", [&]() {
                                    AttentionSoftmaxCUDABwdImpl<scalar_t>(
                                        *(y_grad.expect_contiguous()),
                                        *(y.expect_contiguous()),
                                        use_causal_mask, x_grad);
                                  });

  return x_grad;
}

}  // namespace ops
}  // namespace megalodon
