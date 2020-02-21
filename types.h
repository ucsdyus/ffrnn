#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace ffrnn {

using NeighborList = std::vector<int>;
using WeightList = std::vector<at::Tensor>;
using NearNeighbor = std::pair<NeighborList, WeightList>;

const float PI = std::atan(1.0f) * 4.0f;
const int KERNEL_SIZE = 4;

#define __hd__ __host__ __device__

}  // namespace ffrnn

#endif  // TYPES_H_