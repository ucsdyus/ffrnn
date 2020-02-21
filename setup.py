from setuptools import setup
from torch.utils import cpp_extension


setup(name='ffrnn',
      ext_modules=[cpp_extension.CppExtension(
          'ffrnn', ['ffrnn.cc'],
          extra_compile_args=["/O2", "/w"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
