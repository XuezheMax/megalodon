#include "blas.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/complex.h>
#include <cublasLt.h>

#include <type_traits>

namespace megalodon {
namespace blas {

namespace {

constexpr cublasOperation_t ToCuBLASOp(TransposeOp op) {
  switch (op) {
    case TransposeOp::kN: {
      return CUBLAS_OP_N;
    }
    case TransposeOp::kT: {
      return CUBLAS_OP_T;
    }
    case TransposeOp::kC: {
      return CUBLAS_OP_C;
    }
    default: {
      TORCH_CHECK(false);
    }
  }
}

}  // namespace

template <>
void GemmCUDA<float>(cublasHandle_t handle, TransposeOp transa,
                     TransposeOp transb, int64_t m, int64_t n, int64_t k,
                     float alpha, const float* a, int64_t lda, const float* b,
                     int64_t ldb, float beta, float* c, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasSgemm(handle, ToCuBLASOp(transb),
                                   ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
                                   a, lda, &beta, c, ldc));
}

template <>
void GemmCUDA<double>(cublasHandle_t handle, TransposeOp transa,
                      TransposeOp transb, int64_t m, int64_t n, int64_t k,
                      double alpha, const double* a, int64_t lda,
                      const double* b, int64_t ldb, double beta, double* c,
                      int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasDgemm(handle, ToCuBLASOp(transb),
                                   ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
                                   a, lda, &beta, c, ldc));
}

template <>
void GemmCUDA<at::Half>(cublasHandle_t handle, TransposeOp transa,
                        TransposeOp transb, int64_t m, int64_t n, int64_t k,
                        float alpha, const at::Half* a, int64_t lda,
                        const at::Half* b, int64_t ldb, float beta, at::Half* c,
                        int64_t ldc) {
  TORCH_CUDABLAS_CHECK(
      cublasGemmEx(handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
                   &alpha, b, CUDA_R_16F, ldb, a, CUDA_R_16F, lda, &beta, c,
                   CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmCUDA<at::BFloat16>(cublasHandle_t handle, TransposeOp transa,
                            TransposeOp transb, int64_t m, int64_t n, int64_t k,
                            float alpha, const at::BFloat16* a, int64_t lda,
                            const at::BFloat16* b, int64_t ldb, float beta,
                            at::BFloat16* c, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(
      cublasGemmEx(handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
                   &alpha, b, CUDA_R_16BF, ldb, a, CUDA_R_16BF, lda, &beta, c,
                   CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmBatchedCUDA<float>(cublasHandle_t handle, TransposeOp transa,
                            TransposeOp transb, int64_t batch_size, int64_t m,
                            int64_t n, int64_t k, float alpha,
                            const float** a_array, int64_t lda,
                            const float** b_array, int64_t ldb, float beta,
                            float** c_array, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasSgemmBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b_array,
      ldb, a_array, lda, &beta, c_array, ldc, batch_size));
}

template <>
void GemmStridedBatchedCUDA<float>(cublasHandle_t handle, TransposeOp transa,
                                   TransposeOp transb, int64_t batch_size,
                                   int64_t m, int64_t n, int64_t k, float alpha,
                                   const float* a, int64_t lda,
                                   int64_t batch_stride_a, const float* b,
                                   int64_t ldb, int64_t batch_stride_b,
                                   float beta, float* c, int64_t ldc,
                                   int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
      batch_stride_b, a, lda, batch_stride_a, &beta, c, ldc, batch_stride_c,
      batch_size));
}

template <>
void GemmBatchedCUDA<double>(cublasHandle_t handle, TransposeOp transa,
                             TransposeOp transb, int64_t batch_size, int64_t m,
                             int64_t n, int64_t k, double alpha,
                             const double** a_array, int64_t lda,
                             const double** b_array, int64_t ldb, double beta,
                             double** c_array, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasDgemmBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b_array,
      ldb, a_array, lda, &beta, c_array, ldc, batch_size));
}

template <>
void GemmStridedBatchedCUDA<double>(cublasHandle_t handle, TransposeOp transa,
                                    TransposeOp transb, int64_t batch_size,
                                    int64_t m, int64_t n, int64_t k,
                                    double alpha, const double* a, int64_t lda,
                                    int64_t batch_stride_a, const double* b,
                                    int64_t ldb, int64_t batch_stride_b,
                                    double beta, double* c, int64_t ldc,
                                    int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
      batch_stride_b, a, lda, batch_stride_a, &beta, c, ldc, batch_stride_c,
      batch_size));
}

template <>
void GemmBatchedCUDA<at::Half>(cublasHandle_t handle, TransposeOp transa,
                               TransposeOp transb, int64_t batch_size,
                               int64_t m, int64_t n, int64_t k, float alpha,
                               const at::Half** a_array, int64_t lda,
                               const at::Half** b_array, int64_t ldb,
                               float beta, at::Half** c_array, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha,
      reinterpret_cast<const void**>(b_array), CUDA_R_16F, ldb,
      reinterpret_cast<const void**>(a_array), CUDA_R_16F, lda, &beta,
      reinterpret_cast<void**>(c_array), CUDA_R_16F, ldc, batch_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmStridedBatchedCUDA<at::Half>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha,
    const at::Half* a, int64_t lda, int64_t batch_stride_a, const at::Half* b,
    int64_t ldb, int64_t batch_stride_b, float beta, at::Half* c, int64_t ldc,
    int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b,
      CUDA_R_16F, ldb, batch_stride_b, a, CUDA_R_16F, lda, batch_stride_a,
      &beta, c, CUDA_R_16F, ldc, batch_stride_c, batch_size, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmBatchedCUDA<at::BFloat16>(cublasHandle_t handle, TransposeOp transa,
                                   TransposeOp transb, int64_t batch_size,
                                   int64_t m, int64_t n, int64_t k, float alpha,
                                   const at::BFloat16** a_array, int64_t lda,
                                   const at::BFloat16** b_array, int64_t ldb,
                                   float beta, at::BFloat16** c_array,
                                   int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha,
      reinterpret_cast<const void**>(b_array), CUDA_R_16BF, ldb,
      reinterpret_cast<const void**>(a_array), CUDA_R_16BF, lda, &beta,
      reinterpret_cast<void**>(c_array), CUDA_R_16BF, ldc, batch_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmStridedBatchedCUDA<at::BFloat16>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha,
    const at::BFloat16* a, int64_t lda, int64_t batch_stride_a,
    const at::BFloat16* b, int64_t ldb, int64_t batch_stride_b, float beta,
    at::BFloat16* c, int64_t ldc, int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b,
      CUDA_R_16BF, ldb, batch_stride_b, a, CUDA_R_16BF, lda, batch_stride_a,
      &beta, c, CUDA_R_16BF, ldc, batch_stride_c, batch_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmStridedBatchedCUDA<c10::complex<float>>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    c10::complex<float> alpha, const c10::complex<float>* a, int64_t lda,
    int64_t batch_stride_a, const c10::complex<float>* b, int64_t ldb,
    int64_t batch_stride_b, c10::complex<float> beta, c10::complex<float>* c,
    int64_t ldc, int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasCgemm3mStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
      reinterpret_cast<const cuComplex*>(&alpha),
      reinterpret_cast<const cuComplex*>(b), ldb, batch_stride_b,
      reinterpret_cast<const cuComplex*>(a), lda, batch_stride_a,
      reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc, batch_stride_c, batch_size));
}

template <>
void GemmStridedBatchedCUDA<c10::complex<double>>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    c10::complex<double> alpha, const c10::complex<double>* a, int64_t lda,
    int64_t batch_stride_a, const c10::complex<double>* b, int64_t ldb,
    int64_t batch_stride_b, c10::complex<double> beta, c10::complex<double>* c,
    int64_t ldc, int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
      reinterpret_cast<const cuDoubleComplex*>(&alpha),
      reinterpret_cast<const cuDoubleComplex*>(b), ldb, batch_stride_b,
      reinterpret_cast<const cuDoubleComplex*>(a), lda, batch_stride_a,
      reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc, batch_stride_c, batch_size));
}

}  // namespace blas
}  // namespace megalodon
