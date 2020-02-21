from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob

include_dir = os.path.dirname(os.path.abspath(__file__))
cc_sources = glob.glob(os.path.join(include_dir, "", "*.cc"))
cu_sources = glob.glob(os.path.join(include_dir, "", "*.cu"))

print("cc sources", cc_sources)
print("cu sources", cu_sources)


setup(name="FFRNN",
      ext_modules=[CUDAExtension(
          "ffrnn", sources=cc_sources + cu_sources,
          extra_compile_args={
              "cxx": ["/O2", "/w", "/std:c++14"],
              "nvcc": ["-O3", "--ptxas-options=-v", "-w", "--std=c++14"]})],
      cmdclass={"build_ext": BuildExtension})
