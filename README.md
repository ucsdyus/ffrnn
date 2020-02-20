# FFRNN for CConv

## Third-party
* Pybind11
    * [Install Stackoverflow](https://stackoverflow.com/questions/54704599/how-to-apt-instal-python-pybind11)
    * [Example](https://github.com/tdegeus/pybind11_examples/tree/master/01_py-list_cpp-vector)
    * [Official Doc](https://pybind11.readthedocs.io/en/stable/basics.html)
* Pytorch
    * [libtorch](https://pytorch.org/cppdocs/installing.html)
    * [C++ Extentions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
    * [Tensor Basics](https://pytorch.org/cppdocs/notes/tensor_basics.html)
    * [Tensor C++ template](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/templates/TensorBody.h)
* cuBlas [**TODO**]
    * Run cuBlas in device code ([forum](https://devtalk.nvidia.com/default/topic/902074/call-cublas-api-from-kernel/?offset=3))
    * Get current cuBlas handle in Pytorch ([reference](https://pytorch.org/cppdocs/api/function_namespaceat_1_1cuda_1a948de5eae6a160bb7d99c81a37db548c.html#exhale-function-namespaceat-1-1cuda-1a948de5eae6a160bb7d99c81a37db548c))


## Notice
* Import torch before importing our library (torch runtime required)    

## Useful Command
1.  ```cmake -DCMAKE_PREFIX_PATH=~/3rdpart/libtorch ..```