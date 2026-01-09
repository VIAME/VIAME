# VIAME CMake Docker Configuration
# Settings for Docker container builds
#
# Usage: cmake -C viame_cmake_base.cmake -C viame_cmake_docker.cmake ...

# Docker-specific settings
set(VIAME_FIXUP_BUNDLE OFF CACHE BOOL "Fixup bundle for distribution")
set(VIAME_VERSION_RELEASE ON CACHE BOOL "Version release build")

# Use system Python in Docker
set(VIAME_BUILD_PYTHON_FROM_SOURCE OFF CACHE BOOL "Build Python from source")

# PyTorch settings for Docker
set(VIAME_BUILD_LIMIT_NINJA ON CACHE BOOL "Disable Ninja for PyTorch")

# Typically disabled in Docker builds
set(VIAME_ENABLE_DIVE OFF CACHE BOOL "Enable DIVE")
set(VIAME_ENABLE_ITK OFF CACHE BOOL "Enable ITK")
set(VIAME_ENABLE_VIVIA OFF CACHE BOOL "Enable ViViA")
