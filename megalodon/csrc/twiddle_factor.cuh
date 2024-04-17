#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

#include <cmath>

#include "twiddle_factor_lut.cuh"

namespace megalodon {
namespace cuda_utils {

template <typename T>
__inline__ __device__ c10::complex<T> TwiddleFactor(int n, int k) {
  constexpr T kPi = T(M_PI);
  const T theta = static_cast<T>(k) / static_cast<T>(n) * kPi;
  T s;
  T c;
  c10::cuda::compat::sincos(theta, &s, &c);
  return c10::complex<T>(c, s);
}

template <>
__inline__ __device__ c10::complex<float> TwiddleFactor(int n, int k) {
  switch (n) {
    case 1: {
      return c10::complex<float>(1.0f, 0.0f);
    }
    case 2: {
      return k == 0 ? c10::complex<float>(1.0f, 0.0f)
                    : c10::complex<float>(0.0f, 1.0f);
    }
    case 4: {
      return c10::complex<float>(
          (k == 3 ? -twiddle_factor_lut::kTwiddleFactorLut4[1][0]
                  : twiddle_factor_lut::kTwiddleFactorLut4[k][0]),
          (k == 3 ? twiddle_factor_lut::kTwiddleFactorLut4[1][1]
                  : twiddle_factor_lut::kTwiddleFactorLut4[k][1]));
    }
    case 8: {
      return c10::complex<float>(
          (k > 4 ? -twiddle_factor_lut::kTwiddleFactorLut8[8 - k][0]
                 : twiddle_factor_lut::kTwiddleFactorLut8[k][0]),
          (k > 4 ? twiddle_factor_lut::kTwiddleFactorLut8[8 - k][1]
                 : twiddle_factor_lut::kTwiddleFactorLut8[k][1]));
    }
    case 16: {
      return c10::complex<float>(
          (k > 8 ? -twiddle_factor_lut::kTwiddleFactorLut16[16 - k][0]
                 : twiddle_factor_lut::kTwiddleFactorLut16[k][0]),
          (k > 8 ? twiddle_factor_lut::kTwiddleFactorLut16[16 - k][1]
                 : twiddle_factor_lut::kTwiddleFactorLut16[k][1]));
    }
    default: {
      constexpr float kPi = float(M_PI);
      const float theta = static_cast<float>(k) / static_cast<float>(n) * kPi;
      float s;
      float c;
      __sincosf(theta, &s, &c);
      return c10::complex<float>(c, s);
    }
  }
}

template <>
__inline__ __device__ c10::complex<double> TwiddleFactor(int n, int k) {
  const double x = static_cast<double>(k) / static_cast<double>(n);
  double s;
  double c;
  sincospi(x, &s, &c);
  return c10::complex<double>(c, s);
}

}  // namespace cuda_utils
}  // namespace megalodon
