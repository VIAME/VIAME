# VIAME CMake CPU-Only Configuration
# Settings for CPU-only builds (no CUDA)
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_desktop.cmake -C viame_cmake_cpu.cmake ...

# Disable CUDA
set(VIAME_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA")
set(VIAME_ENABLE_CUDNN OFF CACHE BOOL "Enable cuDNN")

# Disable CUDA-dependent features
set(VIAME_ENABLE_LEARN OFF CACHE BOOL "Enable learning/training")
set(VIAME_ENABLE_PYTORCH-PYSOT OFF CACHE BOOL "Enable PySoT")
set(VIAME_ENABLE_PYTORCH-ULTRALYTICS OFF CACHE BOOL "Enable Ultralytics")
