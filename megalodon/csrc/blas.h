#pragma once

#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace megalodon {
namespace blas {

// https://github.com/pytorch/pytorch/issues/73328
constexpr int64_t kWorkspaceSize = (1 << 20);

enum class TransposeOp {
  kN = 0,
  kT = 1,
  kC = 2,
};

template <typename T>
void GemmCUDA(cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
              int64_t m, int64_t n, int64_t k, at::opmath_type<T> alpha,
              const T* a, int64_t lda, const T* b, int64_t ldb,
              at::opmath_type<T> beta, T* c, int64_t ldc);

template <typename T>
void GemmBatchedCUDA(cublasHandle_t handle, TransposeOp transa,
                     TransposeOp transb, int64_t batch_size, int64_t m,
                     int64_t n, int64_t k, at::opmath_type<T> alpha,
                     const T** a_array, int64_t lda, const T** b_array,
                     int64_t ldb, at::opmath_type<T> beta, T** c_array,
                     int64_t ldc);

template <typename T>
void GemmStridedBatchedCUDA(cublasHandle_t handle, TransposeOp transa,
                            TransposeOp transb, int64_t batch_size, int64_t m,
                            int64_t n, int64_t k, at::opmath_type<T> alpha,
                            const T* a, int64_t lda, int64_t batch_stride_a,
                            const T* b, int64_t ldb, int64_t batch_stride_b,
                            at::opmath_type<T> beta, T* c, int64_t ldc,
                            int64_t batch_stride_c);

}  // namespace blas
}  // namespace megalodon
