# VIAME CMake Desktop Configuration
# Settings for standalone desktop/workstation builds
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_desktop.cmake ...

# Desktop-specific settings
set(VIAME_FIXUP_BUNDLE ON CACHE BOOL "Fixup bundle for distribution")

# Build Python from source for portability
set(VIAME_BUILD_PYTHON_FROM_SOURCE ON CACHE BOOL "Build Python from source")
set(VIAME_PYTHON_VERSION "3.12.12" CACHE STRING "Python version")

# PyTorch settings
set(VIAME_BUILD_LIMIT_NINJA OFF CACHE BOOL "Disable Ninja for PyTorch")

# Desktop applications
set(VIAME_ENABLE_DIVE ON CACHE BOOL "Enable DIVE")

# Additional features for desktop
set(VIAME_ENABLE_PYTORCH-LEARN ON CACHE BOOL "Enable learning/training")
set(VIAME_ENABLE_POSTGRESQL ON CACHE BOOL "Enable PostgreSQL")

# Model downloads for desktop
set(VIAME_DOWNLOAD_MODELS-DEFAULT-FISH ON CACHE BOOL "Download fish models")
set(VIAME_DOWNLOAD_MODELS-GENERIC ON CACHE BOOL "Download generic models")
