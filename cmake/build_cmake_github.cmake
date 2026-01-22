# VIAME CMake GitHub Actions Configuration
# Settings specific to GitHub Actions release builds
#
# This file contains overrides for CI/CD builds on GitHub Actions.
# Include this file LAST to override settings from other preset files.
#
# Usage: cmake -C build_cmake_base.cmake -C build_cmake_desktop.cmake -C build_cmake_linux.cmake -C build_cmake_github.cmake ...

# Disable model downloads for release builds (models are downloaded separately)
set(VIAME_DOWNLOAD_MODELS OFF CACHE BOOL "Disable model downloads for GitHub release builds")

# Disable VIVIA for release builds
set(VIAME_ENABLE_VIVIA OFF CACHE BOOL "Disable VIVIA for GitHub release builds")
