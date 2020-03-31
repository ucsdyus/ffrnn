from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
import os
import glob

include_dir = os.path.dirname(os.path.abspath(__file__))
cc_sources = glob.glob(os.path.join(include_dir, "", "*.cc"))
cu_sources = glob.glob(os.path.join(include_dir, "", "*.cu"))

print("cc sources", cc_sources)
print("cu sources", cu_sources)


setup(name="FFRNN",
      ext_modules=[CUDAExtension(
          "ffrnn", sources=["ffrnn.cc", "ffrnn_gpu.cu"],
          extra_compile_args={
              "cxx": ["/O2", "/w"],
            #   "cxx": ["-O3", "-w", "-std=c++11"],
              "nvcc": ["-O3", "--ptxas-options=-v", "-w"]})],
      cmdclass={"build_ext": BuildExtension})

setup(name="FFRNN_TEST",
      ext_modules=[CppExtension(
          "ffrnn_test", sources=["ffrnn_test.cc", "ffrnn_cpu.cc", "transform.cc"],
          extra_compile_args={
              "cxx": ["/O2", "/w"],
            #   "cxx": ["-O3", "-w", "-std=c++11"],
            })],
      cmdclass={"build_ext": BuildExtension})
