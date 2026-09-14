#! /bin/bash

# Activate python environment
source /Users/kitware/miniconda3/bin/activate

# Update VIAME sub git sources
export VIAME_SOURCE_FOLDER=/Users/kitware/Jenkins/workspace/VIAME-MacOS-CPU-Release
export VIAME_BUILD_FOLDER=${VIAME_SOURCE_FOLDER}/build
export VIAME_INSTALL_FOLDER=${VIAME_BUILD_FOLDER}/install

cd $VIAME_SOURCE_FOLDER
git submodule update --init --recursive
mkdir $VIAME_BUILD_FOLDER
cd $VIAME_BUILD_FOLDER

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$PATH:$VIAME_INSTALL_FOLDER/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIAME_INSTALL_FOLDER/lib

# Configure VIAME
cmake -S .. -B . --preset macos

# Build VIAME first attempt
make -j$(nproc) -k || true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# HACK: Copy mac python environment to installer
cp -r /Users/kitware/miniconda3 install

# HACK: Double tap the build tree
# Should be removed when non-determinism in kwiver python build fixed
make -j$(nproc)

# HACK: Copy setup_viame.sh.install over setup_viame.sh
# Should be removed when this issue is fixed
cp ../cmake/setup_viame.sh.install install/setup_viame.sh

# HACK: Ensure invalid libsvm symlink isn't created
# Should be removed when this issue is fixed
rm install/lib/libsvm.so
cp install/lib/libsvm.so.2 install/lib/libsvm.so

