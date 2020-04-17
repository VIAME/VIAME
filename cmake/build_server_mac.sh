#! /bin/bash

# Update VIAME sub git sources
export VIAME_SOURCE_FOLDER=/home/kitware/workspace/VIAME_release_macos
export VIAME_BUILD_FOLDER=${VIAME_SOURCE_FOLDER}/build
export VIAME_INSTALL_FOLDER=${VIAME_BUILD_FOLDER}/install

cd $VIAME_SOURCE_FOLDER
git submodule update --init --recursive
mkdir $VIAME_BUILD_FOLDER
cd $VIAME_BUILD_FOLDER

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$PATH:$VIAME_INSTALL_FOLDER/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIAME_INSTALL_FOLDER/lib:$VIAME_INSTALL_FOLDER/lib/python3.6
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$VIAME_INSTALL_FOLDER/include/python3.6m
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$VIAME_INSTALL_FOLDER/include/python3.6m

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_CREATE_PACKAGE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CAFFE:BOOL=OFF \
-DVIAME_ENABLE_CAMTRAWL:BOOL=ON \
-DVIAME_ENABLE_CUDA:BOOL=OFF \
-DVIAME_ENABLE_CUDNN:BOOL=OFF \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_GDAL:BOOL=ON \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=OFF \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_ENABLE_PYTHON-INTERNAL:BOOL=ON \
-DVIAME_ENABLE_PYTORCH:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-INTERNAL:BOOL=OFF \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=OFF \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=OFF \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=OFF \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL_TK:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_YOLO:BOOL=ON 

# Build VIAME first attempt
make -j$(nproc) -k || true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

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
