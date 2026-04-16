# VIAME CMake Linux-Specific Configuration
# Settings specific to Linux desktop builds
#
# Usage: cmake -C build_cmake_base.cmake -C build_cmake_desktop.cmake -C build_cmake_linux.cmake ...

# Build DIVE desktop client from source (requires Node.js 18+ and yarn)
set(VIAME_BUILD_DIVE_FROM_SOURCE ON CACHE BOOL "Build DIVE from source")

# Always enable the VIAME test suite so build scripts can run the CRITICAL
# ctest set after build. Test sources are referenced via CMAKE_CURRENT_SOURCE_DIR
# and are not installed, so this does not affect the contents of the tarball.
set(VIAME_ENABLE_TESTS ON CACHE BOOL "Build VIAME tests")
