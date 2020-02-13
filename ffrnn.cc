  
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

#include <torch/torch.h>


void hello_cpp_torch() {
    std::cout << "Hello Torch" << std::endl;
    at::Tensor mat = torch::rand({3,3});
    at::Tensor identity = torch::ones({3,3});
    std::cout << mat << std::endl;
    std::cout << mat * identity << std::endl;
}


namespace py = pybind11;

PYBIND11_MODULE(ffrnn, m)
{
  m.doc() = "Fast Fixed-radius Nearest Neighbor";

  m.def("hello", &hello_cpp_torch, "Hello World");
}