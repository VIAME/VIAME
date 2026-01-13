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

# CUDA support
set(VIAME_ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(VIAME_ENABLE_CUDNN ON CACHE BOOL "Enable cuDNN")

# Core libraries - always enabled
set(VIAME_ENABLE_KWIVER ON CACHE BOOL "Enable KWIVER")
set(VIAME_ENABLE_VXL ON CACHE BOOL "Enable VXL")
set(VIAME_ENABLE_OPENCV ON CACHE BOOL "Enable OpenCV")
set(VIAME_OPENCV_VERSION "4.9.0" CACHE STRING "OpenCV version")

# Video/codec support
set(VIAME_ENABLE_FFMPEG ON CACHE BOOL "Enable FFmpeg")
set(VIAME_ENABLE_FFMPEG-X264 ON CACHE BOOL "Enable x264 codec")

# Python support
set(VIAME_ENABLE_PYTHON ON CACHE BOOL "Enable Python")

# PyTorch support
set(VIAME_ENABLE_PYTORCH ON CACHE BOOL "Enable PyTorch")
set(VIAME_BUILD_PYTORCH_FROM_SOURCE ON CACHE BOOL "Build PyTorch from source")
set(VIAME_BUILD_TORCHVISION_FROM_SOURCE ON CACHE BOOL "Build TorchVision")
set(VIAME_PYTORCH_VERSION "2.7.1" CACHE STRING "PyTorch version")
set(VIAME_ENABLE_PYTORCH-MMDET ON CACHE BOOL "Enable MMDetection")
set(VIAME_ENABLE_PYTORCH-NETHARN ON CACHE BOOL "Enable Netharn")
set(VIAME_ENABLE_PYTORCH-VISION ON CACHE BOOL "Enable PyTorch Vision")
set(VIAME_ENABLE_PYTORCH-SIAMMASK ON CACHE BOOL "Enable SiamMask")
set(VIAME_ENABLE_PYTORCH-SAM2 ON CACHE BOOL "Enable SAM2")
set(VIAME_ENABLE_PYTORCH-ULTRALYTICS ON CACHE BOOL "Enable Ultralytics")
set(VIAME_ENABLE_PYTORCH-MIT-YOLO ON CACHE BOOL "Enable MIT-YOLO")
set(VIAME_ENABLE_PYTORCH-RF-DETR ON CACHE BOOL "Enable RF-DETR")

# Other frameworks
set(VIAME_ENABLE_DARKNET ON CACHE BOOL "Enable Darknet")

# Typically disabled features
set(VIAME_ENABLE_FLASK OFF CACHE BOOL "Enable Flask")
set(VIAME_ENABLE_GDAL OFF CACHE BOOL "Enable GDAL")
set(VIAME_ENABLE_MATLAB OFF CACHE BOOL "Enable MATLAB")
set(VIAME_ENABLE_SEAL OFF CACHE BOOL "Enable SEAL")
set(VIAME_ENABLE_TENSORFLOW OFF CACHE BOOL "Enable TensorFlow")
