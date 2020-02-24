# FFRNN for CConv

## Third-party
* Pybind11
    * [Install Stackoverflow](https://stackoverflow.com/questions/54704599/how-to-apt-instal-python-pybind11): No Longer Need, PyTorch self included
    * [Example](https://github.com/tdegeus/pybind11_examples/tree/master/01_py-list_cpp-vector)
    * [Official Doc](https://pybind11.readthedocs.io/en/stable/basics.html)
* Pytorch
    * [libtorch install](https://pytorch.org/cppdocs/installing.html): No Longer Need, Use setuptool and virtualenv instead
    * [pytorch setuptools](https://pytorch.org/docs/stable/cpp_extension.html)
        * Not initialization error: constexpr => const
        * pybind cast incomplete pointer error : *(this->value) => *((type *)(this->value)) [link](https://github.com/pytorch/pytorch/issues/11004)
    * [C++ Extentions](https://pytorch.org/tutorials/advanced/cpp_extension.html): Use setuptool insteas of Cmake
    * [Python Extentions](https://pytorch.org/docs/stable/notes/extending.html)
    * [Tensor Basics](https://pytorch.org/cppdocs/notes/tensor_basics.html)
    * [Tensor C++ template](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/templates/TensorBody.h)
* Windows
    * [chocolety](https://chocolatey.org/install)
    * cuDNN [download](https://developer.nvidia.com/rdp/cudnn-download), [install-zh](https://blog.csdn.net/cmat2/article/details/80407059)
* cuBlas [**TODO**]
    * Run cuBlas in device code ([forum](https://devtalk.nvidia.com/default/topic/902074/call-cublas-api-from-kernel/?offset=3))
    * Get current cuBlas handle in Pytorch ([reference](https://pytorch.org/cppdocs/api/function_namespaceat_1_1cuda_1a948de5eae6a160bb7d99c81a37db548c.html#exhale-function-namespaceat-1-1cuda-1a948de5eae6a160bb7d99c81a37db548c))


## Notice
* Import torch before importing our library (torch runtime required)    

## Useful Command on Windows
1. ```Get-ExecutionPolicy```
2. ```Set-ExecutionPolicy -ExecutionPolicy Unrestricted``` / ```Set-ExecutionPolicy -ExecutionPolicy Restricted```
3. ```virtualenv NAME --system-site-packages```
4. ```NAME/Scripts/activate```
5. ```deactivate```
6. ```python setup.py install```
7. ```git push -u origin NewBranchName```
