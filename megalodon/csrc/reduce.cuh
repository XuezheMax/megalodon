#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <limits>

#include "cuda_utils.cuh"
#include "welford.h"

namespace megalodon {
namespace reduce {

template <typename T>
struct SumOp {
  static constexpr T kIdentityElement = T(0);

  __inline__ __device__ T operator()(T lhs, T rhs) const { return lhs + rhs; }
};

template <typename T>
struct SumOp<utils::WelfordData<T>> {
  static constexpr utils::WelfordData<T> kIdentityElement = {};

  __inline__ __device__ utils::WelfordData<T> operator()(
      utils::WelfordData<T> lhs, utils::WelfordData<T> rhs) const {
    return lhs + rhs;
  }
};

template <typename T>
struct MaxOp {
  static constexpr T kIdentityElement = std::numeric_limits<T>::lowest();

  __inline__ __device__ T operator()(T lhs, T rhs) const {
    return c10::cuda::compat::max(lhs, rhs);
  }
};

template <typename T>
struct MinOp {
  static constexpr T kIdentityElement = std::numeric_limits<T>::max();

  __inline__ __device__ T operator()(T lhs, T rhs) const {
    return c10::cuda::compat::min(lhs, rhs);
  }
};

// WarpReduce and WarpAllReduce are adapted from cuda_utils::WarpReduce in
// PyTorch.
// https://github.com/pytorch/pytorch/blob/46a25cc0db0a47aec560f732a2228c319ca8a589/aten/src/ATen/native/cuda/block_reduce.cuh#L114
//
// BlockReduce and BlockAllReduce are adapted from cuda_utils::BlockReduce in
// PyTorch
// https://github.com/pytorch/pytorch/blob/46a25cc0db0a47aec560f732a2228c319ca8a589/aten/src/ATen/native/cuda/block_reduce.cuh#L124
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

// https://github.com/pytorch/pytorch/blob/46a25cc0db0a47aec560f732a2228c319ca8a589/aten/src/ATen/native/cuda/block_reduce.cuh#L114
template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T WarpReduce(T x, ReduceOp reduce_op = SumOp<T>()) {
#pragma unroll
  for (int64_t offset = (cuda_utils::kWarpSize >> 1); offset > 0;
       offset >>= 1) {
    x = reduce_op(x, cuda_utils::WarpShflDown(x, offset));
  }
  return x;
}

// https://github.com/pytorch/pytorch/blob/46a25cc0db0a47aec560f732a2228c319ca8a589/aten/src/ATen/native/cuda/block_reduce.cuh#L124
template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T BlockReduce(T x, T* shm,
                                    ReduceOp reduce_op = SumOp<T>()) {
  if (blockDim.x == cuda_utils::kWarpSize) {
    return WarpReduce(x, reduce_op);
  }
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % cuda_utils::kWarpSize;
  const int64_t wid = tid / cuda_utils::kWarpSize;
  const int64_t num_warps = blockDim.x / cuda_utils::kWarpSize;
  x = WarpReduce(x, reduce_op);
  __syncthreads();
  if (lid == 0) {
    shm[wid] = x;
  }
  __syncthreads();
  x = tid < num_warps ? shm[tid] : ReduceOp::kIdentityElement;
  if (wid == 0) {
    x = WarpReduce(x, reduce_op);
  }
  return x;
}

template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T WarpAllReduce(T x, ReduceOp reduce_op = SumOp<T>()) {
#pragma unroll
  for (int64_t offset = (cuda_utils::kWarpSize >> 1); offset > 0;
       offset >>= 1) {
    x = reduce_op(x, cuda_utils::WarpShflXor(x, offset));
  }
  return x;
}

template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T BlockAllReduce(T x, T* shm,
                                       ReduceOp reduce_op = SumOp<T>()) {
  if (blockDim.x == cuda_utils::kWarpSize) {
    return WarpAllReduce(x, reduce_op);
  }
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % cuda_utils::kWarpSize;
  const int64_t wid = tid / cuda_utils::kWarpSize;
  const int64_t num_warps = blockDim.x / cuda_utils::kWarpSize;
  x = WarpReduce(x, reduce_op);
  __syncthreads();
  if (lid == 0) {
    shm[wid] = x;
  }
  __syncthreads();
  x = lid < num_warps ? shm[lid] : ReduceOp::kIdentityElement;
  x = WarpAllReduce(x, reduce_op);
  return x;
}

}  // namespace reduce
}  // namespace megalodon
