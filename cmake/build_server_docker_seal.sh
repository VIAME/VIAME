#!/bin/bash

# VIAME Docker Seal Build Script

# debugging flag
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

# Fletch, VIAME, CMAKE system deps
install_system_deps apt

# Use GCC 13 for the build. Ubuntu 20.04 ships GCC 9.x, but the PyTorch 2.12
# source build requires GCC >= 11.3; gcc-13 also avoids a gcc-12 false-positive
# -Wmaybe-uninitialized in AVX512 intrinsics that breaks fbgemm's -Werror build
# and is accepted by CUDA 12.6's nvcc.
setup_gcc_toolset 13

# Use Python 3.10 for the build. The Docker build uses the system interpreter
# (build_cmake_docker.cmake sets VIAME_BUILD_PYTHON_FROM_SOURCE=OFF), but Ubuntu
# 20.04 ships Python 3.8 and the from-source PyTorch 2.12 build requires >= 3.10.
# Install Python 3.10 from the deadsnakes PPA and bootstrap pip for it; the cmake
# configure below is pointed at it via -DPython_EXECUTABLE.
export DEBIAN_FRONTEND=noninteractive
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# Install CMAKE
install_cmake

# Update VIAME sub git deps
update_git_submodules /viame
setup_build_directory /viame

# Add VIAME and CUDA paths to build
setup_basic_build_environment /viame/build/install /usr/local/cuda

# Configure VIAME using cache presets
cmake ../ \
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_docker.cmake \
  -DPython_EXECUTABLE:PATH=/usr/bin/python3.10 \
  -DVIAME_ENABLE_ITK:BOOL=ON \
  -DVIAME_ENABLE_WEB_EXCLUDES:BOOL=ON \
  -DVIAME_ENABLE_PYTORCH-LEARN:BOOL=OFF \
  -DVIAME_ENABLE_PYTORCH-SIAMMASK:BOOL=ON \
  -DVIAME_ENABLE_PYTORCH-ULTRALYTICS:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-DEFAULT-FISH:BOOL=OFF \
  -DVIAME_DOWNLOAD_MODELS-ARCTIC-SEAL:BOOL=ON

# Perform multi-threaded build
run_build build_log.txt true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# Fix libsvm symlink issue
fix_libsvm_symlink install

# Finalize Docker install
finalize_docker_install /viame/build
