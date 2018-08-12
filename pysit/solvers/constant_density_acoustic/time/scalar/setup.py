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
                           library_dirs=['./'],
                           language="c++",
                           sources=["cython_wrapper.pyx"],
                           include_dirs=[numpy.get_include()])],
)
