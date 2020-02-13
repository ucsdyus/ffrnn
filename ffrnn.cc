  
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


inline bool in_radius(at::Tensor r) {
    return torch::dot(r, r).item<float>() <= 1.0f;
}

inline void ball2cyl(float x, float y, float z, float &rx, float &ry, float &rz) {
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


inline void cyl2cube(float x, float y, float z, float &rx, float &ry, float &rz) {
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

inline at::Tensor ball2cube(at::Tensor r) {
    float* data_ptr = r.data_ptr<float>();
    float rx, ry, rz;
    float h[] = {0, 0, 0};

    ball2cyl(data_ptr[0], data_ptr[1], data_ptr[2], rx, ry, rz);
    cyl2cube(rx, ry, rz, h[0], h[1], h[2]);
    h[0] = (h[0] + 1.0f) / 2.0f;
    h[1] = (h[1] + 1.0f) / 2.0f;
    h[2] = (h[2] + 1.0f) / 2.0f;
    
    return torch::from_blob(h, {3}, r.options());
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
                at::Tensor hij = ball2cube(r);
                near_neighbor[i].first.push_back(j);
                near_neighbor[i].second.push_back(hij);
                
                at::Tensor hji = ball2cube(-r);
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

  m.def("ball2cube", &ball2cube, "Translate a ball into a cube");

  m.def("bf_cpu_frnn", &bf_cpu_frnn, "Brual Forch CPU ver");
}