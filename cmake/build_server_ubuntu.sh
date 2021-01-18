#! /bin/bash

# debugging flag
set -x

# Fletch, VIAME, CMAKE system deps
apt-get update 
apt-get install -y zip \
git \
wget \
curl \
libcurl4-openssl-dev \
libgl1-mesa-dev \
libexpat1-dev \
libgtk2.0-dev \
libxt-dev \
libxml2-dev \
liblapack-dev \
libx264-dev \
openssl \
libssl-dev \
g++ \
zlib1g-dev \
bzip2 \
libbz2-dev \
liblzma-dev


# Install CMAKE
wget https://cmake.org/files/v3.17/cmake-3.17.0.tar.gz
tar zxvf cmake-3.*
cd cmake-3.17.0
./bootstrap --prefix=/usr/local --system-curl
make -j$(nproc)
make install
cd /
rm -rf cmake-3.17.0.tar.gz

# Update VIAME sub git deps
cd /viame/
git submodule update --init --recursive
mkdir build
cd build 

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$PATH:/viame/build/install/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/viame/build/install/lib:/viame/build/install/lib/python3.6
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/viame/build/install/include/python3.6m
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/viame/build/install/include/python3.6m

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_CREATE_PACKAGE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CAFFE:BOOL=OFF \
-DVIAME_ENABLE_CAMTRAWL:BOOL=ON \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
-DVIAME_ENABLE_FASTER_RCNN:BOOL=OFF \
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
-DVIAME_ENABLE_PYTORCH-INTERNAL:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=ON \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL_TK:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_DARKNET:BOOL=ON 

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

# HACK: Copy in CUDA dlls missed by create_package
# Should be removed when this issue is fixed
cp -P /usr/local/cuda/lib64/libcudart.so* install/lib
cp -P /usr/local/cuda/lib64/libcusparse.so* install/lib
cp -P /usr/local/cuda/lib64/libcufft.so* install/lib
cp -P /usr/local/cuda/lib64/libcusolver.so* install/lib
cp -P /usr/local/cuda/lib64/libnvrtc* install/lib
cp -P /usr/local/cuda/lib64/libnvToolsExt.so* install/lib

# HACK: Copy in CUDNN dlls missing by create_package
# Should be removed when this issue is fixed
cp -P /usr/lib/x86_64-linux-gnu/libcudnn*so.7* install/lib
rm install/lib/libcudnn.so || true
ln -s libcudnn.so.7 install/lib/libcudnn.so

# HACK: Copy in other possible library requirements if present
# Should be removed when this issue is fixed
cp /lib/x86_64-linux-gnu/libreadline.so.6 install/lib || true
cp /lib/x86_64-linux-gnu/libreadline.so.7 install/lib || true
cp /lib/x86_64-linux-gnu/libpcre.so.3 install/lib || true
cp /lib/x86_64-linux-gnu/libexpat.so.1 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libcrypto.so install/lib || true
cp /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libfreetype.so.6 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libx264.so.152 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libgomp.so.1 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libSM.so.6 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libICE.so.6 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libblas.so.3 install/lib || true
cp /usr/lib/x86_64-linux-gnu/liblapack.so.3 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libgfortran.so.3 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libgfortran.so.4 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libquadmath.so.0 install/lib || true

cp -P /usr/lib/x86_64-linux-gnu/libnccl.so* install/lib || true

# HACK: Okay these are a bit much
cp /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libpng16.so.16 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 install/lib || true
cp /usr/lib/x86_64-linux-gnu/libgraphite2.so.3 install/lib || true

# HACK: Copy in ubuntu 18.04 specific libraries
source /etc/lsb-release

if [ "${DISTRIB_DESCRIPTION}" == "Ubuntu 18.04.3 LTS" ]; then
    wget https://data.kitware.com/api/v1/item/5e2cdcbbaf2e2eed353a323e/download
    mv download download.tar.gz
    tar -xvf download.tar.gz
    rm download.tar.gz
fi
