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
set(VIAME_ENABLE_PYTORCH-SAM2 ON CACHE BOOL "Enable PyTorch SAM2")
set(VIAME_ENABLE_PYTORCH-SAM3 ON CACHE BOOL "Enable PyTorch SAM3")
set(VIAME_ENABLE_PYTORCH-STEREO ON CACHE BOOL "Enable PyTorch Stereo")
set(VIAME_ENABLE_PYTORCH-RF-DETR ON CACHE BOOL "Enable PyTorch RF-DETR")
set(VIAME_ENABLE_PYTORCH-MIT-YOLO ON CACHE BOOL "Enable PyTorch MIT YOLO")
set(VIAME_ENABLE_PYTORCH-HUGGINGFACE ON CACHE BOOL "Enable PyTorch HuggingFace")

# Desktop applications
set(VIAME_ENABLE_DIVE ON CACHE BOOL "Enable DIVE")
set(VIAME_ENABLE_VIVIA ON CACHE BOOL "Enable ViViA")

# Additional features for desktop
set(VIAME_ENABLE_PYTORCH-LEARN ON CACHE BOOL "Enable learning/training")
set(VIAME_ENABLE_ONNX ON CACHE BOOL "Enable ONNX")
set(VIAME_ENABLE_POSTGRESQL ON CACHE BOOL "Enable PostgreSQL")

# Model downloads for desktop
set(VIAME_DOWNLOAD_MODELS-DEFAULT-FISH ON CACHE BOOL "Download fish models")
set(VIAME_DOWNLOAD_MODELS-GENERIC ON CACHE BOOL "Download generic models")
set(VIAME_DOWNLOAD_MODELS-SRNN ON CACHE BOOL "Download SRNN models")
