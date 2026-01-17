# coding: UTF-8
from setuptools import setup

# TODO:
# - Wrap learning.
# - Make LabelCompatibility, UnaryEnergy, PairwisePotential extensible? (Maybe overkill?)


# If Cython is available, build using Cython.
# Otherwise, use the pre-built (by someone who has Cython, i.e. me) wrapper `.cpp` files.
import os
import sys
import sysconfig

# Find Python's include directory for Python.h
include_dirs = ["pydensecrf/densecrf/include"]

def find_python_include():
    """Find Python.h include directory using multiple methods."""
    # Method 1: Check relative to executable (for embedded/portable Python on Windows)
    exe_dir = os.path.dirname(sys.executable)
    for rel_path in ['../include', 'include', '../../include']:
        candidate = os.path.normpath(os.path.join(exe_dir, rel_path))
        if os.path.exists(os.path.join(candidate, 'Python.h')):
            return candidate

    # Method 2: Check sysconfig
    python_include = sysconfig.get_path('include')
    if python_include and os.path.exists(os.path.join(python_include, 'Python.h')):
        return python_include

    # Method 3: Check sys.prefix
    prefix_include = os.path.join(sys.prefix, 'include')
    if os.path.exists(os.path.join(prefix_include, 'Python.h')):
        return prefix_include

    # Method 4: Check sys.base_prefix (for virtual environments)
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        base_include = os.path.join(sys.base_prefix, 'include')
        if os.path.exists(os.path.join(base_include, 'Python.h')):
            return base_include

    return None

python_include_dir = find_python_include()
if python_include_dir:
    include_dirs.append(python_include_dir)

eigen_include = os.environ.get('EIGEN_INCLUDE_DIR')
if eigen_include:
    include_dirs.append(eigen_include)

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
