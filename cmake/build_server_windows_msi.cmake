# VIAME Windows GPU MSI Build Platform Configuration
# Minimal build configuration for MSI installer
#
# Uses shared cmake preset files for common settings

# CTest configuration
set(CTEST_SITE "noctae.kitware.com")
set(CTEST_BUILD_NAME "Windows_GPU_MSI_Main")
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

# Include base cmake preset file only (MSI build is minimal)
include_cmake_preset(build_cmake_base.cmake)

# Windows-specific CUDA paths (uses CUDA 12.6)
add_option("CUDA_NVCC_EXECUTABLE:PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
add_option("CUDNN_ROOT_DIR:PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

# Windows-specific build paths
add_option("VIAME_BUILD_KWIVER_DIR" "C:/tmp/kv5")
add_option("VIAME_BUILD_PLUGINS_DIR" "C:/tmp/vm5")

# MSI-specific settings - minimal build
add_option("VIAME_FIXUP_BUNDLE" "ON")
add_option("VIAME_ENABLE_CUDA" "ON")
add_option("VIAME_ENABLE_CUDNN" "ON")
add_option("VIAME_ENABLE_DIVE" "OFF")
add_option("VIAME_ENABLE_FFMPEG-X264" "OFF")
add_option("VIAME_ENABLE_LEARN" "OFF")
add_option("VIAME_ENABLE_ONNX" "OFF")
add_option("VIAME_ENABLE_POSTGRESQL" "OFF")
add_option("VIAME_ENABLE_PYTORCH" "OFF")
add_option("VIAME_ENABLE_PYTORCH-MMDET" "OFF")
add_option("VIAME_ENABLE_PYTORCH-NETHARN" "OFF")
add_option("VIAME_ENABLE_VIVIA" "OFF")
add_option("VIAME_PYTHON_BUILD_FROM_SOURCE" "ON")
add_option("VIAME_DOWNLOAD_MODELS" "ON")

# Finalize OPTIONS variable
finalize_options()

set(platform Windows10)
