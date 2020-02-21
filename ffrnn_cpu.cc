#include <vector>
#include <cmath>
#include <torch/extension.h>

#include "transform.h"
#include "types.h"
#include "util_tools.h"

namespace ffrnn {

namespace {

inline bool in_radius(at::Tensor r) {
    return torch::dot(r, r).item<float>() <= 1.0f;
}

}  // namespace

std::vector<NearNeighbor> bf_cpu(at::Tensor points, float R) {
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

}  // namespace ffrnn
