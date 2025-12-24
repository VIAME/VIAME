#!/bin/bash

# VIAME Rocky Linux CPU-Only Build Script

# Debugging, logging, and options
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

export VIAME_SOURCE_DIR=/viame

# Extract version from RELEASE_NOTES.md
extract_viame_version $VIAME_SOURCE_DIR
export VIAME_BUILD_DIR=$VIAME_SOURCE_DIR/build
export VIAME_INSTALL_DIR=$VIAME_BUILD_DIR/install

# Install system dependencies and use more recent compiler
install_system_deps yum

# Install more modern CMAKE and OpenSSL from source
install_openssl
install_cmake

# Use GCC11 for build
yum install -y gcc-toolset-11
source /opt/rh/gcc-toolset-11/enable

# Update VIAME sub git sources
update_git_submodules $VIAME_SOURCE_DIR
setup_build_directory $VIAME_SOURCE_DIR

# Configure Paths [should be removed when no longer necessary by fletch]
setup_build_environment $VIAME_INSTALL_DIR "" "3.10"

# Configure VIAME using cache presets
cmake ../ \
  -C ../cmake/viame_cmake_base.cmake \
  -C ../cmake/viame_cmake_desktop.cmake \
  -C ../cmake/viame_cmake_cpu.cmake

# Build VIAME and setup libraries, pipe output to file
run_build_and_setup_libraries > build_log.txt 2>&1

# Verify build success and create tarball
if verify_build_success build_log.txt; then
  create_install_tarball "CPU-$VIAME_VERSION" "Linux-64Bit"
else
  exit 1
fi
