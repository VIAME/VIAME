# coding: UTF-8
from setuptools import setup

# TODO:
# - Wrap learning.
# - Make LabelCompatibility, UnaryEnergy, PairwisePotential extensible? (Maybe overkill?)


# If Cython is available, build using Cython.
# Otherwise, use the pre-built (by someone who has Cython, i.e. me) wrapper `.cpp` files.
import os
eigen_include = os.environ.get('EIGEN_INCLUDE_DIR')
if eigen_include:
    include_dirs = ["pydensecrf/densecrf/include", eigen_include]
else:
    include_dirs = ["pydensecrf/densecrf/include"]

from setuptools.extension import Extension

# Define extensions with proper include directories
eigen_ext = Extension(
    "pydensecrf.eigen",
    ["pydensecrf/eigen.pyx", "pydensecrf/eigen_impl.cpp"],
    language="c++",
    include_dirs=include_dirs
)

densecrf_ext = Extension(
    "pydensecrf.densecrf",
    ["pydensecrf/densecrf.pyx",
     "pydensecrf/densecrf/src/densecrf.cpp",
     "pydensecrf/densecrf/src/unary.cpp",
     "pydensecrf/densecrf/src/pairwise.cpp",
     "pydensecrf/densecrf/src/permutohedral.cpp",
     "pydensecrf/densecrf/src/optimization.cpp",
     "pydensecrf/densecrf/src/objective.cpp",
     "pydensecrf/densecrf/src/labelcompatibility.cpp",
     "pydensecrf/densecrf/src/util.cpp",
     "pydensecrf/densecrf/external/liblbfgs/lib/lbfgs.c"],
    language="c++",
    include_dirs=include_dirs + ["pydensecrf/densecrf/external/liblbfgs/include"]
)

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([eigen_ext, densecrf_ext])
except ImportError:
    # Fallback to pre-built .cpp files if Cython not available
    eigen_ext.sources = ["pydensecrf/eigen.cpp", "pydensecrf/eigen_impl.cpp"]
    densecrf_ext.sources[0] = "pydensecrf/densecrf.cpp"
    ext_modules = [eigen_ext, densecrf_ext]

setup(
    name="pydensecrf",
    version="1.0",
    description="A python interface to Philipp Krähenbühl's fully-connected (dense) CRF code.",
    long_description="See the README.md at http://github.com/lucasb-eyer/pydensecrf",
    author="Lucas Beyer",
    author_email="lucasb.eyer.be@gmail.com",
    url="http://github.com/lucasb-eyer/pydensecrf",
    ext_modules=ext_modules,
    packages=["pydensecrf"],
    setup_requires=['cython==0.29.36'],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
