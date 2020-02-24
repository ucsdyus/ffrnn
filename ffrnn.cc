  
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cassert>
#include <torch/extension.h>

#include "types.h"
#include "transform.h"

namespace ffrnn {

NnList_t bf_cpu(torch::Tensor points, float R);

}  // namespace ffrnn

namespace py = pybind11;

PYBIND11_MODULE(ffrnn, m) {
    m.doc() = "Fast Fixed-radius Nearest Neighbor";

    // Transform
    m.def("th_ball2cube", &ffrnn::th_ball2cube, "Translate a ball into a cube");

    m.def("th_weighted_ball2grid", &ffrnn::th_weighted_ball2grid, 
    "Translate a ball into grid with trilinear interpolation and smooth weights.");

    m.def("th_ball2grid", &ffrnn::th_ball2grid, "Translate a ball into grid with trilinear interpolation");

    m.def("th_ball2grid_with_window", &ffrnn::th_ball2grid_with_window, 
    "Translate a ball into grid with trilinear interpolation and window weights.");

    // FFRNN
    m.def("bf_cpu", &ffrnn::bf_cpu, "Brual Forch CPU ver");
}