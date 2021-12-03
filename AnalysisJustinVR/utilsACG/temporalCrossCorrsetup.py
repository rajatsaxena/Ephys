# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("temporalCrossCorrelogram", ["temporalCrossCorrelogram.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-fopenmp'],
      	extra_link_args=['-lgomp']
    )]
setup(
    name = "Temporal AutoCorrelogram",
    ext_modules = cythonize(extensions),  # accepts a glob pattern
)