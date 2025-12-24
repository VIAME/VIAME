#! /bin/bash

# Stage 2: Build remaining components (kwiver, pytorch-libs, viame)
# This script continues from the Stage 1 artifact which contains fletch and pytorch builds.
# It builds all remaining components and creates the final release package.

set -x
set -e

export VIAME_SOURCE_DIR=/viame

# Extract version from RELEASE_NOTES.md (first token of first line)
export VIAME_VERSION=$(head -n 1 $VIAME_SOURCE_DIR/RELEASE_NOTES.md | awk '{print $1}')
export VIAME_BUILD_DIR=$VIAME_SOURCE_DIR/build
export VIAME_INSTALL_DIR=$VIAME_BUILD_DIR/install

export CUDA_DIRECTORY=/usr/local/cuda-viame
export CUDNN_DIRECTORY=/usr

# Install system dependencies and use more recent compiler
$VIAME_SOURCE_DIR/cmake/build_server_deps_yum.sh

# Install more modern CMAKE and OpenSSL from source
./viame/cmake/build_server_linux_ssl.sh
./viame/cmake/build_server_linux_cmake.sh

# Patch CUDNN when required
./viame/cmake/build_server_patch_cudnn.sh

# Use GCC11 for build
yum install -y gcc-toolset-11
source /opt/rh/gcc-toolset-11/enable

# Hack for storing paths to CUDA libs for some libraries
rm -f /usr/local/cuda
rm -f /usr/local/cuda-12
mv /usr/local/cuda-12.6 $CUDA_DIRECTORY

# Update VIAME sub git sources (needed for source files not in Stage 1 artifact)
echo "Checking out VIAME submodules"

cd $VIAME_SOURCE_DIR
git config --global --add safe.directory $VIAME_SOURCE_DIR
git submodule update --init --recursive

# Build directory should already exist from Stage 1 artifact
cd build

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$VIAME_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$VIAME_INSTALL_DIR/lib:$VIAME_INSTALL_DIR/lib/python3.10:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$CPLUS_INCLUDE_PATH

# Build Stage 2: Everything else (kwiver, pytorch-libs, viame)
echo "Beginning Stage 2 build (kwiver, pytorch-libs, viame), routing build info to build_log_stage2.txt"

# Run full build - CMake will skip already-built targets from Stage 1
../cmake/build_server_linux_build.sh $CUDA_DIRECTORY > build_log_stage2.txt 2>&1

# Output check statements
if grep -q "Built target viame" build_log_stage2.txt; then
  echo "VIAME Build Succeeded"

  # Make tarball of install
  mv install viame
  rm VIAME-${VIAME_VERSION}-Linux-64Bit.tar.gz ||:
  tar -zcvf VIAME-${VIAME_VERSION}-Linux-64Bit.tar.gz viame
  mv viame install
else
  echo "VIAME Build Failed"
  exit 1
fi

if grep -q "fixup_bundle: preparing..." build_log_stage2.txt; then
  echo "Fixup Bundle Called Successfully"
else
  echo "Error: Fixup Bundle Not Called"
  exit 1
fi

echo "Stage 2 build completed successfully - Final release package created"
