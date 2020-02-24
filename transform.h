#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace ffrnn {

extern inline void ball2cyl(float x, float y, float z, float& rx, float& ry, float& rz);

extern inline void cyl2cube(float x, float y, float z, float& rx, float& ry, float& rz);

extern inline void ball2cube(float* r_ptr, float* h_ptr);

extern inline void weighted_ball2grid(float* r_ptr, float* grid_ptr, float smooth_weight);

extern inline void ball2grid_with_window(float* r_ptr, float* grid_ptr);

extern inline torch::Tensor th_ball2cube(torch::Tensor r);

extern inline torch::Tensor th_ball2grid(torch::Tensor r);

extern inline torch::Tensor th_weighted_ball2grid(torch::Tensor r, float smooth_weight);

extern inline torch::Tensor th_ball2grid_with_window(torch::Tensor r);

}  // namespace ffrnn

#endif  // TRANSFORM_H_
