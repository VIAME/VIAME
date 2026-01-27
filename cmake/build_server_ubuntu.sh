#!/bin/bash

# VIAME Ubuntu Build Script

# debugging flag
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

# Extract version from RELEASE_NOTES.md
extract_viame_version /viame

# System Deps
install_system_deps apt

# Install CMAKE
install_cmake

# Install Node.js and yarn for DIVE desktop build
install_nodejs_and_yarn 18

# Update VIAME sub git deps
update_git_submodules /viame
setup_build_directory /viame

# Configure Paths [should be removed when no longer necessary by fletch]
setup_basic_build_environment /viame/build/install

source ./viame/cmake/linux_add_internal_py36_paths.sh

# Configure VIAME using cache presets
cmake ../ \
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_desktop.cmake \
  -C ../cmake/build_cmake_linux.cmake

# Build VIAME, pipe output to file
run_build build_log.txt

# Verify build success
if verify_build_success build_log.txt; then
  prepare_linux_desktop_install install /viame
  create_install_tarball "$VIAME_VERSION" "Ubuntu-64Bit"
else
  exit 1
fi
