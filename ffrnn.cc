  
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

constexpr const float PI = 3.14159265359f;
constexpr const int kernel_size = 4;


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
    float rt_x2_y2_z2 = sqrt(x2_y2_z2);
    if (5.0f / 4.0f * z2 <= x2_y2) {
        float p = rt_x2_y2_z2 / sqrt(x2_y2);
        rx = x * p;
        ry = y * p;
        rz = 3.0f / 2.0f * z;
    } else {
        float p = sqrt(3.0f * rt_x2_y2_z2 / (rt_x2_y2_z2 + abs(z)));
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
    float rt_x2_y2 = sqrt(x * x + y * y);
    if (abs(y) <= abs(x)) {
        if (x > 0) rx = rt_x2_y2;
        else rx = -rt_x2_y2;

        ry = rx * atan2(y, x) * 4.0f / PI;
        rz = z;
        
    } else {
        if (y > 0) ry = rt_x2_y2;
        else ry = -rt_x2_y2;

        rx = ry * atan2(x, y) * 4.0f / PI;
        rz = z;
    }
}

inline void ball2cube(float* r_ptr, float* h_ptr) {
    float rx, ry, rz;
    ball2cyl(r_ptr[0], r_ptr[1], r_ptr[2], rx, ry, rz);
    cyl2cube(rx, ry, rz, h_ptr[0], h_ptr[1], h_ptr[2]);
    h_ptr[0] = (h_ptr[0] + 1.0f) / 2.0f;
    h_ptr[1] = (h_ptr[1] + 1.0f) / 2.0f;
    h_ptr[2] = (h_ptr[2] + 1.0f) / 2.0f;
}

at::Tensor th_ball2cube(at::Tensor r) {
    at::Tensor grid = torch::zeros_like(r);
    ball2cube(r.data_ptr<float>(), grid.data_ptr<float>());
    return grid;
}

inline bool inside(int i, int j, int k) {
    return i >= 0 && j >= 0 && k >= 0 && i < kernel_size && j < kernel_size && k < kernel_size;
}

inline float trilinear_w(float d, int b) {
    return b * d + (1 - b) * (1 - d);
}

at::Tensor th_ball2gird(at::Tensor r) {
    // kernel_size = kernel_size * kernel_size * kernel_size;

    float h[3];
    ball2cube(r.data_ptr<float>(), h);
    std::cout << "h: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    h[0] *= kernel_size;
    h[1] *= kernel_size;
    h[2] *= kernel_size;
    
    at::Tensor grid = torch::zeros(kernel_size * kernel_size * kernel_size, r.options());
    float* grid_data = grid.data_ptr<float>();
    int idx[3] = {static_cast<int>(h[0]), static_cast<int>(h[1]), static_cast<int>(h[2])};
    float dx[3] = {h[0] - idx[0], h[1] - idx[1], h[2] - idx[2]};
    int fi[] = {0, 1, 1, 0, 0, 1, 1, 0};
    int fj[] = {0, 0, 1, 1, 0, 0, 1, 1};
    int fk[] = {0, 0, 0, 0, 1, 1, 1, 1};

    std::cout << "h * size: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    std::cout << "idx[]: " << idx[0] << " " << idx[1] << " " << idx[2] << std::endl;

#pragma unroll
    for (int t = 0; t < 8; ++t) {
        int i = idx[0] + fi[t];
        int j = idx[1] + fj[t];
        int k = idx[2] + fk[t];
        std::cout << i << " " << j << " " << k << std::endl;
        if (!inside(i, j, k)) continue;
        grid_data[i * kernel_size * kernel_size + j * kernel_size + k] = 
            trilinear_w(dx[0], fi[t]) * trilinear_w(dx[1], fj[t]) * trilinear_w(dx[2], fk[t]);;
    }
    return grid;
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
                at::Tensor hij = th_ball2gird(r);
                near_neighbor[i].first.push_back(j);
                near_neighbor[i].second.push_back(hij);
                
                at::Tensor hji = th_ball2gird(-r);
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

  m.def("bf_cpu_frnn", &bf_cpu_frnn, "Brual Forch CPU ver");
}