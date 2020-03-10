#include <vector>
#include <cmath>
#include <torch/extension.h>

#include "transform.h"
#include "types.h"

namespace ffrnn {

namespace {

inline bool in_radius(torch::Tensor r) {
    return torch::dot(r, r).item<float>() <= 1.0f;
}

}  // namespace

std::vector<at::Tensor> bf_cpu(torch::Tensor sources, torch::Tensor candidates, float R, bool include_diag) {
    
    int N = torch::size(sources, 0);
    int M = torch::size(candidates, 0);
    std::vector<std::vector<int>> nn_table(N);
    // std::cout << "Start" << std::endl;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if(include_diag || i != j) {
                torch::Tensor r = (candidates[j] - sources[i]) / R;
                if (in_radius(r)) {
                    nn_table[i].push_back(j);
                }
            }
        }
    }
    // std::cout << "Done Matching." << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     std::cout << i << " neighbor size " << nn_table[i].size() << std::endl;
    // }

    auto th_option = sources.options();
    torch::Tensor th_nn_offset = torch::zeros(
        N + 1, th_option.dtype(torch::kInt32));
    int* nn_offset = th_nn_offset.data_ptr<int>();

    // int max_nnum = 0;
    for (int i = 1; i <= N; ++i) {
        nn_offset[i] = nn_offset[i - 1] + nn_table[i - 1].size();
        
        // max_nnum = std::max(max_nnum, (int) nn_table[i - 1].size());
    }

    // std::cout << "Start Allocating: " << nn_offset[N] << " entries" << std::endl;
    // std::cout << "Avg N Num: " << ((float) nn_offset[N] / (float) N) << " Max N Num: " << max_nnum << std::endl;

    torch::Tensor th_nn_list = torch::zeros(
        nn_offset[N], th_option.dtype(torch::kInt32));
    torch::Tensor th_nw_list = torch::zeros(
        nn_offset[N] * SPATIAL_SIZE, th_option.dtype(torch::kFloat32));
    torch::Tensor th_grad_nn_offset = torch::zeros(
        M + 1, th_option.dtype(torch::kInt32));
    torch::Tensor th_grad_nn_list = torch::zeros(
        nn_offset[N] * 2, th_option.dtype(torch::kInt32));
    
    int* nn_list = th_nn_list.data_ptr<int>();
    float* nw_list = th_nw_list.data_ptr<float>();
    int* grad_nn_offset = th_grad_nn_offset.data_ptr<int>();
    int* grad_nn_list = th_grad_nn_list.data_ptr<int>();

    // std::cout << "Done Allocate " << N << std::endl;
    
    for(int u = 0; u < N; ++u) {
        int Ns = nn_offset[u + 1] - nn_offset[u];
        int* nn = nn_list + nn_offset[u];
        float* nw = nw_list + nn_offset[u] * SPATIAL_SIZE;

        for (int i = 0; i < Ns; ++i) {
            int v = nn_table[u][i];
            nn[i] = v;
            // std::cout << "u, v: " << u << " " << v << std::endl;
            torch::Tensor r = (candidates[v] - sources[u]) / R;
            // std::cout << "r tesnor" << r << " R: " << R << std::endl; 
            ball2grid_with_window(r.data_ptr<float>(), nw + i * SPATIAL_SIZE);
            int start = (nn_offset[v] + grad_nn_offset[v + 1]) * 2;
            grad_nn_list[start] = u;
            grad_nn_list[start + 1] = i;
            ++grad_nn_offset[v + 1];
        }
    }
    for (int i = 1; i <= M; ++i) {
        grad_nn_offset[i] += grad_nn_offset[i - 1];
    }
    // std::cout << "Done transform" << std::endl;
    return {th_nn_offset, th_nn_list, th_nw_list, th_grad_nn_offset, th_grad_nn_list};
}
}  // namespace ffrnn
