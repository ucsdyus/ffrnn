#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace ffrnn {

#define PI 3.14159265359f;
const int KERNEL_SIZE = 4;
const int SPATIAL_SIZE = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE;

#define __hd__ __host__ __device__
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.type().is_cpu(), #x " must be a CPU tensor")

}  // namespace ffrnn

#endif  // TYPES_H_