  
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cassert>
#include <torch/extension.h>

#include "types.h"

namespace ffrnn {

// API:
// sources: N Points which we want to find neighbors for
// candidates: M  Candidate set where we find neighbors
// R: radius
// include_diag: bool
// return: nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list
// nn_offset: N + 1
// nn_list: squeeze(N x Ns)
// nw_lsit: squeeze(N x Ns x Spatial)
// grad_nn_offset: M + 1 (should meet grad_nn_offset[M] = nn_offset[N] = N x Ns)
// grad_nn_lsit: squeeze(N x Ns x 2: <v, v_offset>)

// Example:
// 1. Find neighbors for dynamic points.
//     res = func_name(points, points, R, False)
// 2. Find neighborhoood boundary
//     res = func_name(points, boundary_points, R, True)

std::vector<at::Tensor> bf_gpu(torch::Tensor sources, torch::Tensor candidates, float R, bool include_diag);

}  // namespace ffrnn

namespace py = pybind11;

PYBIND11_MODULE(ffrnn, m) {
    m.doc() = "Fast Fixed-radius Nearest Neighbor";

    // // FFRNN

    m.def("bf_gpu", &ffrnn::bf_gpu, "Brual Forch GPU");
}