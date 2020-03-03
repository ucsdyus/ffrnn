#include "transform.h"
#include "types.h"
#include <cuda.h>
#include <cmath>


namespace ffrnn {

namespace {

__hd__ bool inside_grid(int i, int j, int k) {
    return i >= 0 && j >= 0 && k >= 0 && i < KERNEL_SIZE && j < KERNEL_SIZE && k < KERNEL_SIZE;
}

__hd__ float trilinear_w(float d, int b) {
    return b * d + (1 - b) * (1 - d);
}

__hd__ float window_smooth_weight(float* r) {
    float v = 1.0f - (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    return v * v * v;
}
}  // namespace


__hd__ void ball2cyl(float x, float y, float z, float& rx, float& ry, float& rz) {
    float x2_y2 = x * x + y * y;
    float z2 = z * z;
    float x2_y2_z2 = x2_y2 + z2;
    if (x2_y2_z2 == 0) {
        rx = ry = rz = 0;
        return;
    }
    float rt_x2_y2_z2 = std::sqrt(x2_y2_z2);
    // std::cout << "rt_x2_y2_z2: " << rt_x2_y2_z2 << std::endl;
    if (5.0f / 4.0f * z2 <= x2_y2) {
        // std::cout << "ball2cyl: " << "case 2" << std::endl; 
        float p = rt_x2_y2_z2 / std::sqrt(x2_y2);
        rx = x * p;
        ry = y * p;
        rz = 3.0f / 2.0f * z;
    } else {
        // std::cout << "ball2cyl: " << "case 3" << std::endl;
        float p = std::sqrt((3.0f * rt_x2_y2_z2) / (rt_x2_y2_z2 + std::abs(z)));
        rx = x * p;
        ry = y * p;
        if (z > 0) rz = rt_x2_y2_z2;
        else rz = -rt_x2_y2_z2;
    }
}


__hd__ void cyl2cube(float x, float y, float z, float& rx, float& ry, float& rz) {
    if (x == 0 && y == 0) {
        rx = ry = 0;
        rz = z;
        return;
    }
    float rt_x2_y2 = std::sqrt(x * x + y * y);
    if (std::abs(y) <= std::abs(x)) {
        // std::cout << "cyl2cube: " << "case 2" << std::endl; 
        if (x > 0) {
            rx = rt_x2_y2;
            ry = rx * std::atan2(y, x) * 4.0f / PI;
        } else {
            rx = -rt_x2_y2;
            ry = rx * std::atan2(-y, -x) * 4.0f / PI;
        }

        rz = z;
        
    } else {
        // std::cout << "cyl2cube: " << "case 3" << std::endl; 
        if (y > 0) {
            ry = rt_x2_y2;
            rx = ry * std::atan2(x, y) * 4.0f / PI;
        } else {
            ry = -rt_x2_y2;
            rx = ry * std::atan2(-x, -y) * 4.0f / PI;
        }
        rz = z;
    }
}

__hd__ void ball2cube(float* r_ptr, float* h_ptr) {
    float rx, ry, rz;
    // std::cout << "ball: " << r_ptr[0] << " " << r_ptr[1] << " " << r_ptr[2] << std::endl;
    ball2cyl(r_ptr[0], r_ptr[1], r_ptr[2], rx, ry, rz);
    // std::cout << "cyl: " << rx << " " << ry << " " << rz << std::endl;
    cyl2cube(rx, ry, rz, h_ptr[0], h_ptr[1], h_ptr[2]);
    // std::cout << "cube: " << h_ptr[0] << " " << h_ptr[1] << " " << h_ptr[2] << std::endl;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        h_ptr[i] = (h_ptr[i] + 1.0f) / 2.0f;
    }
}

__hd__ void weighted_ball2grid(float* r_ptr, float* grid_ptr, float smooth_weight) {
    float h[3];
    ball2cube(r_ptr, h);
    // std::cout << "h: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    h[0] *= (KERNEL_SIZE - 1);
    h[1] *= (KERNEL_SIZE - 1);
    h[2] *= (KERNEL_SIZE - 1);
    
    int idx[3];
    float dx[3]; 
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        idx[i] = static_cast<int>(floor(h[i]));
        if (idx[i] == KERNEL_SIZE - 1) --idx[i];
        dx[i] = h[i] - idx[i];
    }

    int fi[] = {0, 1, 1, 0, 0, 1, 1, 0};
    int fj[] = {0, 0, 1, 1, 0, 0, 1, 1};
    int fk[] = {0, 0, 0, 0, 1, 1, 1, 1};

    // std::cout << "h * size: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    // std::cout << "idx[]: " << idx[0] << " " << idx[1] << " " << idx[2] << std::endl;

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int i = idx[0] + fi[t];
        int j = idx[1] + fj[t];
        int k = idx[2] + fk[t];
        // std::cout << i << " " << j << " " << k << std::endl;
        // Safe to remove it.
        // if (!inside_grid(i, j, k)) continue;
        // std::cout << "work" << i << " " << j << " " << k << std::endl;
        grid_ptr[i * KERNEL_SIZE * KERNEL_SIZE + j * KERNEL_SIZE + k] = 
            smooth_weight * trilinear_w(dx[0], fi[t]) *
            trilinear_w(dx[1], fj[t]) * trilinear_w(dx[2], fk[t]);
    }
}

__hd__ void ball2grid_with_window(float* r_ptr, float* grid_ptr) {
    // std::cout << "r_ptr: " << r_ptr[0] << " " << r_ptr[1] << " " << r_ptr[3] << std::endl;
    // std::cout << window_smooth_weight(r_ptr) << std::endl;
    weighted_ball2grid(r_ptr, grid_ptr, window_smooth_weight(r_ptr));
}

__hd__ torch::Tensor th_ball2cube(torch::Tensor r) {
    torch::Tensor cube = torch::zeros_like(r);
    ball2cube(r.data_ptr<float>(), cube.data_ptr<float>());
    return cube;
}

__hd__ torch::Tensor th_weighted_ball2grid(torch::Tensor r, float smooth_weight) {
    torch::Tensor grid = torch::zeros(SPATIAL_SIZE, r.options());
    weighted_ball2grid(r.data_ptr<float>(), grid.data_ptr<float>(), smooth_weight);
    return grid;
}

__hd__ torch::Tensor th_ball2grid(torch::Tensor r) {
    return th_weighted_ball2grid(r, /*smooth_weight=*/1.0f);
}

__hd__ torch::Tensor th_ball2grid_with_window(torch::Tensor r) {
    // std::cout << window_smooth_weight(r.data_ptr<float>()) << std::endl;
    return th_weighted_ball2grid(r, window_smooth_weight(r.data_ptr<float>()));
}
}  // namespace ffrnn
