#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <tuple>

namespace megalodon {
namespace random_utils {

constexpr int64_t kRandomUnroll = 4;

}  // namespace random_utils
}  // namespace megalodon
