# VIAME CMake Base Configuration
# Common defaults shared across all build configurations
#
# Usage: cmake -C /path/to/viame_cmake_base.cmake ...
#   Then override specific options as needed

# Build type
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

# Core VIAME settings
set(VIAME_ENABLE_DOCS OFF CACHE BOOL "Enable documentation")

# CUDA support
set(VIAME_ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(VIAME_ENABLE_CUDNN ON CACHE BOOL "Enable cuDNN")

# Core libraries - always enabled
set(VIAME_ENABLE_OPENCV ON CACHE BOOL "Enable OpenCV")

# Python support
set(VIAME_ENABLE_PYTHON ON CACHE BOOL "Enable Python")

# PyTorch support
set(VIAME_ENABLE_PYTORCH ON CACHE BOOL "Enable PyTorch")
set(VIAME_PYTORCH_VERSION "2.12.0" CACHE STRING "PyTorch version")
set(VIAME_ENABLE_PYTORCH-MMDET ON CACHE BOOL "Enable MMDetection")
set(VIAME_ENABLE_PYTORCH-NETHARN ON CACHE BOOL "Enable Netharn")
set(VIAME_ENABLE_PYTORCH-VISION ON CACHE BOOL "Enable TorchVision")
set(VIAME_ENABLE_PYTORCH-SIAMMASK ON CACHE BOOL "Enable SiamMask")
set(VIAME_ENABLE_PYTORCH-SAM2 ON CACHE BOOL "Enable SAM2")
set(VIAME_ENABLE_PYTORCH-SAM3 ON CACHE BOOL "Enable SAM3")
set(VIAME_ENABLE_PYTORCH-ULTRALYTICS ON CACHE BOOL "Enable Ultralytics")
set(VIAME_ENABLE_PYTORCH-HUGGINGFACE ON CACHE BOOL "Enable HuggingFace")
set(VIAME_ENABLE_PYTORCH-MIT-YOLO ON CACHE BOOL "Enable MIT-YOLO")
set(VIAME_ENABLE_PYTORCH-RF-DETR ON CACHE BOOL "Enable RF-DETR")
set(VIAME_ENABLE_PYTORCH-STEREO ON CACHE BOOL "Enable FF Stereo")
set(VIAME_ENABLE_PYTORCH-LEARN ON CACHE BOOL "Enable LEARN")
set(VIAME_ENABLE_PYTORCH-DINO3 ON CACHE BOOL "Enable DINOv3")

# Other frameworks
set(VIAME_ENABLE_ONNX ON CACHE BOOL "Enable ONNX")
set(VIAME_ENABLE_COLMAP ON CACHE BOOL "Enable COLMAP")
set(VIAME_ENABLE_DARKNET ON CACHE BOOL "Enable Darknet")

# Model packs
set(VIAME_DOWNLOAD_MODELS-SIAMMASK ON CACHE BOOL "Enable SiamMask")
set(VIAME_DOWNLOAD_MODELS-SRNN OFF CACHE BOOL "Download SRNN models")
