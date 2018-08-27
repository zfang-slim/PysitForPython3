from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext
import os
import os.path
#
os.environ["CC"] = "gcc-8"
os.environ["CXX"] = "g++-8"

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("_constant_density_acoustic_time_scalar_cpp",
                           library_dirs=['./', '/usr/local/Cellar/gcc/8.2.0/lib/gcc/8'], #lib for MacOS lib64 for Linux , '/usr/lib'
                           language="c++",
                           sources=["cython_wrapper.pyx"],
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=["-O3","-fopenmp","-ffast-math"],
                           library=["gomp"])]
)
