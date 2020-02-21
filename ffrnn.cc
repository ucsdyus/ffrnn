  
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

#include <torch/extension.h>


void SANITY_CHECK(at::Tensor points) {
    assert(at::size(points, 1) == 3);
} 

using NeighborList = std::vector<int>;
using WeightList = std::vector<at::Tensor>;
using NearNeighbor = std::pair<NeighborList, WeightList>;

const float PI = std::atan(1.0f) * 4.0f;
const int kernel_size = 4;


inline bool in_radius(at::Tensor r) {
    return torch::dot(r, r).item<float>() <= 1.0f;
}

inline void ball2cyl(float x, float y, float z, float& rx, float& ry, float& rz) {
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


inline void cyl2cube(float x, float y, float z, float& rx, float& ry, float& rz) {
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

inline void ball2cube(float* r_ptr, float* h_ptr) {
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

at::Tensor th_ball2cube(at::Tensor r) {
    at::Tensor cube = torch::zeros_like(r);
    ball2cube(r.data_ptr<float>(), cube.data_ptr<float>());
    return cube;
}

inline bool inside_grid(int i, int j, int k) {
    return i >= 0 && j >= 0 && k >= 0 && i < kernel_size && j < kernel_size && k < kernel_size;
}

inline float trilinear_w(float d, int b) {
    return b * d + (1 - b) * (1 - d);
}

inline at::Tensor th_weighted_ball2grid(at::Tensor r, float smooth_weight) {
    
    // kernel_size = kernel_size * kernel_size * kernel_size;

    float h[3];
    ball2cube(r.data_ptr<float>(), h);
    // std::cout << "h: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    h[0] *= (kernel_size - 1);
    h[1] *= (kernel_size - 1);
    h[2] *= (kernel_size - 1);
    
    at::Tensor grid = torch::zeros(kernel_size * kernel_size * kernel_size, r.options());
    float* grid_data = grid.data_ptr<float>();
    int idx[3];
    float dx[3]; 
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        idx[i] = static_cast<int>(floor(h[i]));
        if (idx[i] == kernel_size - 1) --idx[i];
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
        grid_data[i * kernel_size * kernel_size + j * kernel_size + k] = 
            smooth_weight * trilinear_w(dx[0], fi[t]) *
            trilinear_w(dx[1], fj[t]) * trilinear_w(dx[2], fk[t]);
    }
    return grid;
}


inline at::Tensor th_ball2grid(at::Tensor r) {
    return th_weighted_ball2grid(r, /*smooth_weight=*/1.0f);
}

inline float window_smooth_weight(float* r, float R) {
    float v = 1.0f - (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / (R * R);
    return v * v * v;
}

inline at::Tensor th_ball2grid_with_window(at::Tensor r, float R) {
    // std::cout << window_smooth_weight(r.data_ptr<float>(), R) << std::endl;
    return th_weighted_ball2grid(r, window_smooth_weight(r.data_ptr<float>(), R));
}

std::vector<NearNeighbor> bf_cpu_frnn(at::Tensor points, float R) {
    // at::Tensor is not differentiable 
    // torch::Tensor is differentiable
    SANITY_CHECK(points);

    int N = at::size(points, 0);
    std::vector<NearNeighbor> near_neighbor(N);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            at::Tensor r = (points[j] - points[i]) / R;
            if (in_radius(r)) {
                at::Tensor hij = th_ball2grid_with_window(r, R);
                near_neighbor[i].first.push_back(j);
                near_neighbor[i].second.push_back(hij);
                
                at::Tensor hji = th_ball2grid_with_window(-r, R);
                near_neighbor[j].first.push_back(i);
                near_neighbor[j].second.push_back(hji);
            }
        }
    }
    return near_neighbor;
}


namespace py = pybind11;

PYBIND11_MODULE(ffrnn, m)
{
  m.doc() = "Fast Fixed-radius Nearest Neighbor";

  m.def("th_ball2cube", &th_ball2cube, "Translate a ball into a cube");

  m.def("th_weighted_ball2grid", &th_weighted_ball2grid, 
    "Translate a ball into grid with trilinear interpolation and smooth weights.");
  
  m.def("th_ball2grid", &th_ball2grid, "Translate a ball into grid with trilinear interpolation");

  m.def("th_ball2grid_with_window", &th_ball2grid_with_window, 
    "Translate a ball into grid with trilinear interpolation and window weights.");

  m.def("bf_cpu_frnn", &bf_cpu_frnn, "Brual Forch CPU ver");
}