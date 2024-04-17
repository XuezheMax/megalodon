#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include <cmath>

namespace megalodon {
namespace complex_utils {

template <typename T>
__inline__ __device__ c10::complex<T> Mul1i(c10::complex<T> x) {
  return c10::complex<T>(-x.imag(), x.real());
}

template <typename T>
__inline__ __device__ T RealOfProduct(c10::complex<T> lhs,
                                      c10::complex<T> rhs) {
  return lhs.real() * rhs.real() - lhs.imag() * rhs.imag();
}

template <typename T>
__inline__ __device__ c10::complex<T> Exp(c10::complex<T> x) {
  const T e = c10::cuda::compat::exp(x.real());
  T s;
  T c;
  c10::cuda::compat::sincos(x.imag(), &s, &c);
  return c10::complex<T>(c * e, s * e);
}

// template <>
// __inline__ __device__ c10::complex<float> Exp(c10::complex<float> x) {
//   const float e = c10::cuda::compat::exp(x.real());
//   float s;
//   float c;
//   __sincosf(x.imag(), &s, &c);
//   return c10::complex<float>(c * e, s * e);
// }

}  // namespace complex_utils
}  // namespace megalodon
