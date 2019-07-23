#! /bin/bash

# debugging and logging
set -x
 
# install Fletch & VIAME system deps
yum update -y
yum -y groupinstall 'Development Tools'
yum install -y zip \
git \
wget \
openssl \
openssl-devel \
zlib \
zlib-devel \
freeglut-devel \
mesa-libGLU-devel \
lapack-devel \
libXt-devel \
libXmu-devel \
libXi-devel \
expat-devel \
readline-devel \
curl \
curl-devel \
atlas-devel 

# Setup anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh -b
export PATH=$PATH:/root/anaconda3/bin
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

# Update VIAME sub git sources
cd /viame/
git fetch -p
git submodule update --init --recursive
mkdir build
cd build 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/viame/build/install/lib

# HACK: Python3.6 copy so that create_package succeeds
# Should be removed when this issue is fixed
mkdir -p install/lib
cp /root/anaconda3/lib/libpython3.6m.* install/lib

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
-DVIAME_ENABLE_GDAL:BOOL=ON \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=OFF \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_ENABLE_PYTORCH:BOOL=ON \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=ON \
-DVIAME_ENABLE_SEAL_TK:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_YOLO:BOOL=ON 

# Build VIAME
make -j$(nproc)

# HACK: Remove libpython.so files necessary for create_package
# Should be removed when this issue is fixed
rm -r install/lib/libpython*

# HACK: Copy setup_viame.sh.install over setup_viame.sh
# Should be removed when this issue is fixed
cp ../viame/cmake/setup_viame.sh.install install/setup_viame.sh

# HACK: Copy in CUDA dlls missed by create_package
# Should be removed when this issue is fixed
cp -P /usr/local/cuda/lib64/libcudart.so* lib
cp -P /usr/local/cuda/lib64/libcusparse.so* lib
cp -P /usr/local/cuda/lib64/libcufft.so* lib
cp -P /usr/local/cuda/lib64/libcusolver.so* lib
