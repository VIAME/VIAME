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
libssl-dev \
liblapack-dev \
openssl \
libssl-dev \
g++ \
zlib1g-dev 

# Setup anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh -b
source /root/anaconda3/bin/activate
rm -rf Anaconda3-5.2.0-Linux-x86_64.sh

# Install CMAKE
wget https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz
tar zxvf cmake-3.*
cd cmake-3.*
./bootstrap --prefix=/usr/local --system-curl
make -j$(nproc)
make install
cd /
rm -rf cmake-3.14.0.tar.gz

# Update VIAME sub git deps
cd /viame/
git submodule update --init --recursive
mkdir build
cd build 

export PATH=$PATH:/viame/build/install/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/viame/build/install/lib

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_CREATE_PACKAGE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CAFFE:BOOL=OFF \
-DVIAME_ENABLE_CAMTRAWL:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=OFF \
-DVIAME_ENABLE_GDAL:BOOL=ON \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=ON \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_ENABLE_PYTORCH:BOOL=OFF \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL_TK:BOOL=ON \
-DVIAME_ENABLE_SMQTK:BOOL=OFF \
-DVIAME_ENABLE_TENSORFLOW:BOOL=ON \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=OFF \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_YOLO:BOOL=ON \
-DVIAME_DOWNLOAD_MODELS-ARCTIC-SEAL:BOOL=ON

# Build VIAME first attempt
make -j$(nproc) -k || true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# HACK: Python3.6 copy so that create_package succeeds
# Should be removed when this issue is fixed
mkdir -p install/lib
cp /root/anaconda3/lib/libpython3.6m.* install/lib

# HACK: Double tap the build tree after adding in python
# Should be removed when this issue is fixed
make -j$(nproc)

# HACK: Remove libpython.so files necessary for create_package
# Should be removed when this issue is fixed
rm -r install/lib/libpython*

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

# HACK: Copy in CUDNN dlls missing by create_package
# Should be removed when this issue is fixed
cp -P /usr/lib/x86_64-linux-gnu/libcudnn*so.7* install/lib
rm install/lib/libcudnn.so || true
ln -s libcudnn.so.7 install/lib/libcudnn.so

# HACK: Copy in other possible library requirements if present
# Should be removed when this issue is fixed
cp /usr/lib64/libva.so.1 install/lib || true
cp /usr/lib64/libreadline.so.6 install/lib || true
cp /usr/lib64/libdc1394.so.22 install/lib || true
cp /usr/lib64/libcrypto.so.10 install/lib || true
cp /usr/lib64/libpcre.so.1 install/lib || true

cp -P /usr/lib/x86_64-linux-gnu/libnccl.so* install/lib || true
