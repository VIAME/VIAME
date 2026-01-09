# VIAME Windows GPU Build Platform Configuration
#
# Uses shared cmake preset files for common settings

# CTest configuration
set(CTEST_SITE "noctae.kitware.com")
set(CTEST_BUILD_NAME "Windows_GPU_Main")
set(CTEST_SOURCE_DIRECTORY "C:/VIAME-Builds/GPU")
set(CTEST_BINARY_DIRECTORY "C:/VIAME-Builds/GPU/build/")
set(CTEST_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_PROJECT_NAME VIAME)
set(CTEST_BUILD_MODEL "Nightly")
set(CTEST_NIGHTLY_START_TIME "3:00:00 UTC")
set(CTEST_USE_LAUNCHERS 1)
include(CTestUseLaunchers)

# Include helper for building OPTIONS from cmake presets
include(${CMAKE_CURRENT_LIST_DIR}/build_common_functions.cmake)

# Include base cmake preset files
include_cmake_preset(build_cmake_base.cmake)
include_cmake_preset(build_cmake_desktop.cmake)

# Windows-specific CUDA paths
add_option("CUDA_NVCC_EXECUTABLE:PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")
add_option("CUDNN_ROOT_DIR:PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

# Windows-specific build paths
add_option("VIAME_BUILD_FLETCH_DIR" "C:/tmp/fl1")
add_option("VIAME_BUILD_KWIVER_DIR" "C:/tmp/kv1")
add_option("VIAME_BUILD_PLUGINS_DIR" "C:/tmp/vm1")

# Windows-specific overrides
add_option("VIAME_ENABLE_DARKNET" "OFF")
add_option("VIAME_BUILD_PYTORCH_FROM_SOURCE:BOOL" "OFF")

# Finalize OPTIONS variable
finalize_options()

set(platform Windows10)
