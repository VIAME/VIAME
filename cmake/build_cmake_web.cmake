# VIAME CMake Web Configuration
# Additional settings for VIAME-Web builds (use with viame_cmake_docker.cmake)
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_docker.cmake -C viame_cmake_web.cmake ...

# Web-specific settings
set(VIAME_ENABLE_WEB_EXCLUDES ON CACHE BOOL "Exclude desktop-only components")
set(VIAME_ENABLE_LEARN ON CACHE BOOL "Enable learning/training")
set(VIAME_ENABLE_KWANT ON CACHE BOOL "Enable KWANT")
set(VIAME_ENABLE_ONNX ON CACHE BOOL "Enable ONNX")

# PyTorch plugins for web
set(VIAME_PYTORCH_BUILD_TORCHVISION ON CACHE BOOL "Build TorchVision")
set(VIAME_ENABLE_PYTORCH-VISION ON CACHE BOOL "Enable PyTorch Vision")
set(VIAME_ENABLE_PYTORCH-PYSOT ON CACHE BOOL "Enable PySoT")
set(VIAME_ENABLE_PYTORCH-SAM ON CACHE BOOL "Enable SAM")
set(VIAME_ENABLE_PYTORCH-ULTRALYTICS ON CACHE BOOL "Enable Ultralytics")

# Model downloads - minimal for web
set(VIAME_DOWNLOAD_MODELS ON CACHE BOOL "Download models")
set(VIAME_DOWNLOAD_MODELS-GENERIC OFF CACHE BOOL "Download generic models")
set(VIAME_DOWNLOAD_MODELS-FISH OFF CACHE BOOL "Download fish models")
set(VIAME_DOWNLOAD_MODELS-PYSOT OFF CACHE BOOL "Download PySoT models")
set(VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL OFF CACHE BOOL "Download Arctic seal models")
set(VIAME_DOWNLOAD_MODELS-HABCAM OFF CACHE BOOL "Download HabCam models")
set(VIAME_DOWNLOAD_MODELS-MOUSS OFF CACHE BOOL "Download MOUSS models")
