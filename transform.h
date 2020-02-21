#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace ffrnn {

extern inline void ball2cyl(float x, float y, float z, float& rx, float& ry, float& rz);

extern inline void cyl2cube(float x, float y, float z, float& rx, float& ry, float& rz);

extern inline void ball2cube(float* r_ptr, float* h_ptr);

extern inline at::Tensor th_ball2cube(at::Tensor r);

extern inline at::Tensor th_ball2grid(at::Tensor r);

extern inline at::Tensor th_weighted_ball2grid(at::Tensor r, float smooth_weight);

extern inline at::Tensor th_ball2grid_with_window(at::Tensor r, float R);

}  // namespace ffrnn

#endif  // TRANSFORM_H_
