#pragma once

#include <c10/macros/Macros.h>

#include <type_traits>

namespace megalodon {
namespace utils {

// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
template <typename T>
struct WelfordData {
  int64_t m0 = 0;
  T m1 = T(0);
  T m2 = T(0);

  C10_HOST_DEVICE WelfordData<T>& operator+=(T x) {
    ++m0;
    const T coef = T(1) / static_cast<T>(m0);
    const T delta1 = x - m1;
    m1 += coef * delta1;
    const T delta2 = delta1 * (x - m1) - m2;
    m2 += coef * delta2;
    return *this;
  }

  C10_HOST_DEVICE WelfordData<T>& operator+=(WelfordData<T> x) {
    m0 += x.m0;
    const T c2 = m0 == 0 ? T(0) : static_cast<T>(x.m0) / static_cast<T>(m0);
    const T c1 = T(1) - c2;
    const T delta = x.m1 - m1;
    m1 = c1 * m1 + c2 * x.m1;
    m2 = c1 * m2 + c2 * x.m2 + (c1 * delta) * (c2 * delta);
    return *this;
  }
};

template <typename T>
C10_HOST_DEVICE WelfordData<T> operator+(WelfordData<T> m, T x) {
  return m += x;
}

template <typename T>
C10_HOST_DEVICE WelfordData<T> operator+(WelfordData<T> lhs,
                                         WelfordData<T> rhs) {
  return lhs += rhs;
}

}  // namespace utils
}  // namespace megalodon
