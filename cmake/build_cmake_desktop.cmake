# VIAME CMake Desktop Configuration
# Settings for standalone desktop/workstation builds
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_desktop.cmake ...

# Desktop-specific settings
set(VIAME_FIXUP_BUNDLE ON CACHE BOOL "Fixup bundle for distribution")

# Build Python from source for portability
set(VIAME_BUILD_PYTHON_FROM_SOURCE ON CACHE BOOL "Build Python from source")
set(VIAME_PYTHON_VERSION "3.10.4" CACHE STRING "Python version")

# PyTorch settings
set(VIAME_BUILD_LIMIT_NINJA OFF CACHE BOOL "Disable Ninja for PyTorch")
set(VIAME_ENABLE_PYTORCH-ULTRALYTICS ON CACHE BOOL "Enable PyTorch Ultralytics")
set(VIAME_ENABLE_PYTORCH-SIAMMASK ON CACHE BOOL "Enable PyTorch SiamMask")

# Desktop applications
set(VIAME_ENABLE_DIVE ON CACHE BOOL "Enable DIVE")
set(VIAME_ENABLE_VIVIA ON CACHE BOOL "Enable ViViA")

# Additional features for desktop
set(VIAME_ENABLE_PYTORCH-LEARN ON CACHE BOOL "Enable learning/training")
set(VIAME_ENABLE_ONNX ON CACHE BOOL "Enable ONNX")
set(VIAME_ENABLE_POSTGRESQL ON CACHE BOOL "Enable PostgreSQL")
