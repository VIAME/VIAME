# VIAME CMake Base Configuration
# Common defaults shared across all build configurations
#
# Usage: cmake -C /path/to/viame_cmake_base.cmake ...
#   Then override specific options as needed

# Build type
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

# Core VIAME settings
set(VIAME_BUILD_DEPENDENCIES ON CACHE BOOL "Build dependencies")
set(VIAME_ENABLE_DOCS OFF CACHE BOOL "Enable documentation")

# Core libraries - always enabled
set(VIAME_ENABLE_KWIVER ON CACHE BOOL "Enable KWIVER")
set(VIAME_ENABLE_VXL ON CACHE BOOL "Enable VXL")
set(VIAME_ENABLE_OPENCV ON CACHE BOOL "Enable OpenCV")
set(VIAME_OPENCV_VERSION "3.4.0" CACHE STRING "OpenCV version")

# Video/codec support
set(VIAME_ENABLE_FFMPEG ON CACHE BOOL "Enable FFmpeg")
set(VIAME_ENABLE_FFMPEG-X264 ON CACHE BOOL "Enable x264 codec")

# Python support
set(VIAME_ENABLE_PYTHON ON CACHE BOOL "Enable Python")

# PyTorch support
set(VIAME_ENABLE_PYTORCH ON CACHE BOOL "Enable PyTorch")
set(VIAME_PYTORCH_BUILD_FROM_SOURCE ON CACHE BOOL "Build PyTorch from source")
set(VIAME_PYTORCH_VERSION "2.7.1" CACHE STRING "PyTorch version")
set(VIAME_ENABLE_PYTORCH-MMDET ON CACHE BOOL "Enable MMDetection")
set(VIAME_ENABLE_PYTORCH-NETHARN ON CACHE BOOL "Enable Netharn")

# Other ML frameworks
set(VIAME_ENABLE_DARKNET ON CACHE BOOL "Enable Darknet")
set(VIAME_ENABLE_SMQTK ON CACHE BOOL "Enable SMQTK")

# Typically disabled features
set(VIAME_ENABLE_FLASK OFF CACHE BOOL "Enable Flask")
set(VIAME_ENABLE_GDAL OFF CACHE BOOL "Enable GDAL")
set(VIAME_ENABLE_MATLAB OFF CACHE BOOL "Enable MATLAB")
set(VIAME_ENABLE_SCALLOP_TK OFF CACHE BOOL "Enable Scallop TK")
set(VIAME_ENABLE_SEAL OFF CACHE BOOL "Enable SEAL")
set(VIAME_ENABLE_TENSORFLOW OFF CACHE BOOL "Enable TensorFlow")
set(VIAME_ENABLE_UW_PREDICTOR OFF CACHE BOOL "Enable UW Predictor")
