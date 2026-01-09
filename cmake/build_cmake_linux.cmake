# VIAME CMake Linux-Specific Configuration
# Settings specific to Linux desktop builds
#
# Usage: cmake -C build_cmake_base.cmake -C build_cmake_desktop.cmake -C build_cmake_linux.cmake ...

# Build DIVE desktop client from source (requires Node.js 18+ and yarn)
set(VIAME_BUILD_DIVE_FROM_SOURCE ON CACHE BOOL "Build DIVE from source")
