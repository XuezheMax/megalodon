#pragma once

#include <c10/util/llvmMathExtras.h>
#include <torch/torch.h>

#include <cstring>
#include <tuple>
#include <utility>

namespace py = pybind11;

namespace megalodon {
namespace utils {

constexpr int64_t kChunkSize = 16;

// DivUp is adapted from caffe2::math::utils::DivUp in PyTorch.
// https://github.com/pytorch/pytorch/blob/440e4353c7bd30fc4754248be67d90e0f6b1d956/caffe2/utils/math/utils.h#L159
//
// CeilLog2 is copied from at::native::utils in PyTorch.
// https://github.com/pytorch/pytorch/blob/440e4353c7bd30fc4754248be67d90e0f6b1d956/aten/src/ATen/native/cpu/utils.h#L126
//
// RowwiseMoments is adapted from at::native::RowwiseMomentsImpl in PyTorch.
// https://github.com/pytorch/pytorch/blob/440e4353c7bd30fc4754248be67d90e0f6b1d956/aten/src/ATen/native/cpu/moments_utils.h#L113
//
// From PyTorch:
//
// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
//
// From Caffe2:
//
// Copyright (c) 2016-present, Facebook Inc. All rights reserved.
//
// All contributions by Facebook:
// Copyright (c) 2016 Facebook Inc.
//
// All contributions by Google:
// Copyright (c) 2015 Google Inc.
// All rights reserved.
//
// All contributions by Yangqing Jia:
// Copyright (c) 2015 Yangqing Jia
// All rights reserved.
//
// All contributions by Kakao Brain:
// Copyright 2019-2020 Kakao Brain
//
// All contributions by Cruise LLC:
// Copyright (c) 2022 Cruise LLC.
// All rights reserved.
//
// All contributions from Caffe:
// Copyright(c) 2013, 2014, 2015, the respective contributors
// All rights reserved.
//
// All other contributions:
// Copyright(c) 2015, 2016 the respective contributors
// All rights reserved.
//
// Caffe2 uses a copyright model similar to Caffe: each contributor holds
// copyright over their contributions to Caffe2. The project versioning records
// all such contribution and copyright details. If a contributor wants to further
// mark their specific copyright on a particular contribution, they should
// indicate their copyright solely in the commit message of the change when it is
// committed.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// https://github.com/pytorch/pytorch/blob/440e4353c7bd30fc4754248be67d90e0f6b1d956/caffe2/utils/math/utils.h#L159
template <typename T>
T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

// https://github.com/pytorch/pytorch/blob/440e4353c7bd30fc4754248be67d90e0f6b1d956/aten/src/ATen/native/cpu/utils.h#L126
template <typename T>
T CeilLog2(T x) {
  if (x <= 2) {
    return 1;
  }
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<T>(c10::llvm::findLastSet(static_cast<uint64_t>(x) - 1) +
                        1);
}

template <typename T>
std::pair<T, T> Fast2Sum(T a, T b) {
  const T s = a + b;
  const T z = s - a;
  const T t = b - z;
  return std::make_pair(s, t);
}

template <typename T>
std::pair<T, T> KahanAdd(T x, T sum, T c) {
  const T y = x - c;
  const T t = sum + y;
  c = t - sum - y;
  return std::make_pair(t, c);
}

template <typename T>
T Cube(T x) {
  return x * x * x;
}

template <typename T>
std::tuple<int64_t, T, T> WelfordUpdate(int64_t m0, T m1, T m2, T x) {
  ++m0;
  const T coef = T(1) / static_cast<T>(m0);
  const T delta1 = x - m1;
  m1 += coef * delta1;
  const T delta2 = delta1 * (x - m1) - m2;
  m2 += coef * delta2;
  return std::make_tuple(m0, m1, m2);
}

template <typename T>
std::tuple<int64_t, T, T> WelfordCombine(int64_t a_m0, T a_m1, T a_m2,
                                         int64_t b_m0, T b_m1, T b_m2) {
  const int64_t m0 = a_m0 + b_m0;
  const T c1 = m0 == 0 ? T(0) : T(a_m0) / static_cast<T>(m0);
  const T c2 = m0 == 0 ? T(0) : T(b_m0) / static_cast<T>(m0);
  const T delta = b_m1 - a_m1;
  const T m1 = c1 * a_m1 + c2 * b_m1;
  const T m2 = c1 * a_m2 + c2 * b_m2 + (c1 * delta) * (c2 * delta);
  return std::make_tuple(m0, m1, m2);
}

// TODO: Optimize by using Vec256.
// Reference:
// RowwiseMoments is adapted from at::native::RowwiseMomentsImpl in PyTorch.
template <typename T, typename T_ACC>
std::tuple<int64_t, T_ACC, T_ACC> RowwiseMoments(int64_t N, const T* X,
                                                 const bool* padding_mask) {
  const int64_t num_chunks = utils::DivUp(N, utils::kChunkSize);
  const int64_t depth = utils::CeilLog2(num_chunks);

  std::vector<int64_t> m0_stk(depth, 0);
  std::vector<T_ACC> m1_stk(depth, T_ACC(0));
  std::vector<T_ACC> m2_stk(depth, T_ACC(0));
  for (int64_t i = 0; i < num_chunks; ++i) {
    const int64_t l = i * utils::kChunkSize;
    const int64_t r = std::min(l + utils::kChunkSize, N);
    for (int64_t j = l; j < r; ++j) {
      const T_ACC x = static_cast<T_ACC>(X[j]);
      const bool mask = padding_mask != nullptr && padding_mask[j];
      const auto [_, u, v] =
          utils::WelfordUpdate(m0_stk[0], m1_stk[0], m2_stk[0], x);
      m0_stk[0] += mask ? 0 : 1;
      m1_stk[0] = mask ? m1_stk[0] : u;
      m2_stk[0] = mask ? m2_stk[0] : v;
    }

    int64_t cnt = i + 1;
    for (int64_t j = 1; j < depth && (cnt & 1) == 0; ++j) {
      std::tie(m0_stk[j], m1_stk[j], m2_stk[j]) =
          utils::WelfordCombine(m0_stk[j], m1_stk[j], m2_stk[j], m0_stk[j - 1],
                                m1_stk[j - 1], m2_stk[j - 1]);
      m0_stk[j - 1] = 0;
      m1_stk[j - 1] = T_ACC(0);
      m2_stk[j - 1] = T_ACC(0);
      cnt >>= 1;
    }
  }
  for (int64_t i = 1; i < depth; ++i) {
    std::tie(m0_stk[0], m1_stk[0], m2_stk[0]) = utils::WelfordCombine(
        m0_stk[0], m1_stk[0], m2_stk[0], m0_stk[i], m1_stk[i], m2_stk[i]);
  }

  return std::make_tuple(m0_stk[0], m1_stk[0], m2_stk[0]);
}

}  // namespace utils
}  // namespace megalodon
