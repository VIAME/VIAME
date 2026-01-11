# VIAME Windows MSI Build Platform Configuration
# Modular build configuration with optional feature addons
#
# Usage: Set VIAME_MSI_STAGE environment variable before running:
#   Stage 1: base     - fletch + kwiver (CPU only)
#   Stage 2: cuda     - adds CUDA/cuDNN support
#   Stage 3: pytorch  - adds PyTorch + pytorch-libs
#   Stage 4: vivia    - adds VIVIA interface
#   Stage 5: seal     - adds SEAL toolkit
#   Stage 6: dive     - adds DIVE interface

# CTest configuration
set(CTEST_SITE "noctae.kitware.com")
set(CTEST_BUILD_NAME "Windows_MSI_$ENV{VIAME_MSI_STAGE}")
set(CTEST_SOURCE_DIRECTORY "C:/VIAME-Builds/GPU-MSI")
set(CTEST_BINARY_DIRECTORY "C:/VIAME-Builds/GPU-MSI/build/")
set(CTEST_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_PROJECT_NAME VIAME)
set(CTEST_BUILD_MODEL "Nightly")
set(CTEST_NIGHTLY_START_TIME "3:00:00 UTC")
set(CTEST_USE_LAUNCHERS 1)
include(CTestUseLaunchers)

# Include helper for building OPTIONS from cmake presets
include(${CMAKE_CURRENT_LIST_DIR}/build_common_functions.cmake)

# Include base cmake preset file
include_cmake_preset(build_cmake_base.cmake)

# Windows-specific build paths
add_option("VIAME_BUILD_FLETCH_DIR" "C:/tmp/fl5")
add_option("VIAME_BUILD_KWIVER_DIR" "C:/tmp/kv5")
add_option("VIAME_BUILD_PLUGINS_DIR" "C:/tmp/vm5")

# Base settings - everything off initially
add_option("VIAME_FIXUP_BUNDLE" "ON")
add_option("VIAME_BUILD_PYTHON_FROM_SOURCE" "ON")

# Default all features to OFF
set(ENABLE_CUDA OFF)
set(ENABLE_PYTORCH OFF)
set(ENABLE_VIVIA OFF)
set(ENABLE_SEAL OFF)
set(ENABLE_DIVE OFF)
set(ENABLE_MODELS OFF)

# Determine enabled features based on stage (cumulative)
set(STAGE "$ENV{VIAME_MSI_STAGE}")
if(NOT STAGE)
  set(STAGE "base")
endif()

# Stage progression: each stage enables previous stages' features
if(STAGE STREQUAL "dive" OR STAGE STREQUAL "6")
  set(ENABLE_DIVE ON)
  set(STAGE "seal")
endif()

if(STAGE STREQUAL "seal" OR STAGE STREQUAL "5")
  set(ENABLE_SEAL ON)
  set(STAGE "vivia")
endif()

if(STAGE STREQUAL "vivia" OR STAGE STREQUAL "4")
  set(ENABLE_VIVIA ON)
  set(STAGE "pytorch")
endif()

if(STAGE STREQUAL "pytorch" OR STAGE STREQUAL "3")
  set(ENABLE_PYTORCH ON)
  set(ENABLE_MODELS ON)
  set(STAGE "cuda")
endif()

if(STAGE STREQUAL "cuda" OR STAGE STREQUAL "2")
  set(ENABLE_CUDA ON)
endif()

# Apply CUDA settings
if(ENABLE_CUDA)
  add_option("VIAME_ENABLE_CUDA" "ON")
  add_option("VIAME_ENABLE_CUDNN" "ON")
  add_option("CUDA_NVCC_EXECUTABLE:PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
  add_option("CUDNN_ROOT_DIR:PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
else()
  add_option("VIAME_ENABLE_CUDA" "OFF")
  add_option("VIAME_ENABLE_CUDNN" "OFF")
endif()

# Apply PyTorch settings
if(ENABLE_PYTORCH)
  add_option("VIAME_ENABLE_PYTORCH-LEARN" "ON")
  add_option("VIAME_ENABLE_PYTORCH" "ON")
  add_option("VIAME_BUILD_PYTORCH_FROM_SOURCE" "OFF")
  add_option("VIAME_BUILD_TORCHVISION_FROM_SOURCE" "OFF")
  add_option("VIAME_ENABLE_PYTORCH-MMDET" "ON")
  add_option("VIAME_ENABLE_PYTORCH-NETHARN" "ON")
  add_option("VIAME_ENABLE_PYTORCH-SIAMMASK" "ON")
else()
  add_option("VIAME_ENABLE_PYTORCH-LEARN" "OFF")
  add_option("VIAME_ENABLE_PYTORCH" "OFF")
  add_option("VIAME_ENABLE_PYTORCH-MMDET" "OFF")
  add_option("VIAME_ENABLE_PYTORCH-NETHARN" "OFF")
endif()

# Apply VIVIA settings
if(ENABLE_VIVIA)
  add_option("VIAME_ENABLE_VIVIA" "ON")
  add_option("VIAME_ENABLE_GDAL" "ON")
else()
  add_option("VIAME_ENABLE_VIVIA" "OFF")
endif()

# Apply SEAL settings
if(ENABLE_SEAL)
  add_option("VIAME_ENABLE_SEAL" "ON")
else()
  add_option("VIAME_ENABLE_SEAL" "OFF")
endif()

# Apply DIVE settings
if(ENABLE_DIVE)
  add_option("VIAME_ENABLE_DIVE" "ON")
else()
  add_option("VIAME_ENABLE_DIVE" "OFF")
endif()

# Apply model download settings
if(ENABLE_MODELS)
  add_option("VIAME_DOWNLOAD_MODELS" "ON")
else()
  add_option("VIAME_DOWNLOAD_MODELS" "OFF")
endif()

# Features always off for MSI builds
add_option("VIAME_ENABLE_FFMPEG-X264" "OFF")
add_option("VIAME_ENABLE_ONNX" "OFF")
add_option("VIAME_ENABLE_POSTGRESQL" "OFF")

# Finalize OPTIONS variable
finalize_options()

set(platform Windows10)
