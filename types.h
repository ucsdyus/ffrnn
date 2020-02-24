#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace ffrnn {

using NeighborList_t = torch::Tensor;  // Ns
using WeightList_t = torch::Tensor;  // Ns x S
using NearNeighbor_t = std::pair<NeighborList_t, WeightList_t>;
using NnList_t = std::vector<NearNeighbor_t>;

const float PI = std::atan(1.0f) * 4.0f;
const int KERNEL_SIZE = 4;
const int SPATIAL_SIZE = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE;

#define __hd__ __host__ __device__

}  // namespace ffrnn

#endif  // TYPES_H_