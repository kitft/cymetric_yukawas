from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext_modules = [
    Extension(
        "float128_roots",
        ["float128_roots.pyx"],
        extra_compile_args=["-std=c99", "-fPIC", "-O3", "-ffast-math", "-march=native", "-mfpmath=sse", "-msse4", "-lquadmath"],
        extra_link_args=["-lquadmath"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)