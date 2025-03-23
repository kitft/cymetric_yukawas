from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extension = [
    Extension(
        "high_precision_roots",
        ["high_precision_roots.pyx"],
        extra_compile_args=["-O3", "-ffast-math"],
    )
]

setup(
    ext_modules=cythonize(extension),
    include_dirs=[np.get_include()]
)