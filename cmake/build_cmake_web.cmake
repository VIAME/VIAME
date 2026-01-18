# VIAME CMake Web Configuration
# Additional settings for VIAME-Web builds (use with viame_cmake_docker.cmake)
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_docker.cmake -C viame_cmake_web.cmake ...

# Web-specific settings
set(VIAME_ENABLE_WEB_EXCLUDES ON CACHE BOOL "Exclude desktop-only components")
set(VIAME_ENABLE_PYTORCH-LEARN ON CACHE BOOL "Enable learning/training")
set(VIAME_ENABLE_ONNX ON CACHE BOOL "Enable ONNX")

# Model downloads - minimal for web
set(VIAME_DOWNLOAD_MODELS ON CACHE BOOL "Download models")
set(VIAME_DOWNLOAD_MODELS-GENERIC OFF CACHE BOOL "Download generic models")
set(VIAME_DOWNLOAD_MODELS-FISH OFF CACHE BOOL "Download fish models")
set(VIAME_DOWNLOAD_MODELS-PYSOT OFF CACHE BOOL "Download PySoT models")
set(VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL OFF CACHE BOOL "Download Arctic seal models")
set(VIAME_DOWNLOAD_MODELS-HABCAM OFF CACHE BOOL "Download HabCam models")
set(VIAME_DOWNLOAD_MODELS-MOUSS OFF CACHE BOOL "Download MOUSS models")
