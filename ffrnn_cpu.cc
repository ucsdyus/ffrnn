#include <vector>
#include <cmath>
#include <torch/extension.h>

#include "transform.h"
#include "types.h"
#include "util_tools.h"

namespace ffrnn {

namespace {

inline bool in_radius(torch::Tensor r) {
    return torch::dot(r, r).item<float>() <= 1.0f;
}

}  // namespace

NnList_t bf_cpu(torch::Tensor points, float R) {
    // torch::Tensor is not differentiable 
    // torch::Tensor is differentiable
    SANITY_CHECK(points);

    int N = torch::size(points, 0);
    std::vector<std::vector<int>> nn_table(N);

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            torch::Tensor r = (points[j] - points[i]) / R;
            if (in_radius(r)) {
                nn_table[i].push_back(j);
                nn_table[j].push_back(i);
                // std::cout << "i, j: " << i << " " << j << std::endl;
            }
        }
    }

    NnList_t nn_list;
    nn_list.reserve(N);
    auto torch_options = points.options();
    for(int u = 0; u < N; ++u) {
        int Ns = nn_table[u].size();
        NeighborList_t neighbor = torch::zeros(
            Ns, torch_options.dtype(torch::kInt32));
        WeightList_t weight = torch::zeros(
            {Ns, SPATIAL_SIZE}, torch_options.dtype(torch::kFloat32));
        nn_list.push_back({neighbor, weight});

        int* n_ptr = neighbor.data_ptr<int>();
        float* w_ptr = weight.data_ptr<float>();

        for (int i = 0; i < nn_table[u].size(); ++i) {
            int v = nn_table[u][i];
            n_ptr[i] = v;

            // std::cout << "u, v: " << u << " " << v << std::endl;
            torch::Tensor r = (points[v] - points[u]) / R;
            // std::cout << "r tesnor" << r << " R: " << R << std::endl; 
            ball2grid_with_window(r.data_ptr<float>(), w_ptr + i * SPATIAL_SIZE);
        }
    }
    return nn_list;
}

}  // namespace ffrnn
