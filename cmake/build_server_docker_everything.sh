#! /bin/bash

# debugging flag
set -x

# Fletch, VIAME, CMAKE system deps
apt-get update 
apt-get install -y zip \
git \
wget \
tar \
curl \
libcurl4-openssl-dev \
libgl1-mesa-dev \
libexpat1-dev \
libgtk2.0-dev \
libxt-dev \
libxml2-dev \
liblapack-dev \
openssl \
libssl-dev \
g++ \
ninja-build \
zlib1g-dev \
bzip2 \
libbz2-dev \
liblzma-dev

# Install CMAKE
wget https://cmake.org/files/v3.23/cmake-3.23.1.tar.gz
tar zxvf cmake-3.*
cd cmake-3.23.1
./bootstrap --prefix=/usr/local --system-curl
make -j$(nproc)
make install
cd /
rm -rf cmake-3.23.1.tar.gz

# Update VIAME sub git deps
cd /viame/
git submodule update --init --recursive
mkdir build
cd build

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$PATH:/usr/local/cuda/bin:/viame/build/install/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/viame/build/install/lib:/viame/build/install/lib/python3.6
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/viame/build/install/include/python3.6m
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/viame/build/install/include/python3.6m

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=OFF \
-DVIAME_VERSION_RELEASE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=ON \
-DVIAME_ENABLE_CAFFE:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
-DVIAME_ENABLE_DIVE:BOOL=OFF \
-DVIAME_ENABLE_DARKNET:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
-DVIAME_ENABLE_GDAL:BOOL=OFF \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=ON \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_LEARN:BOOL=ON \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_OPENCV_VERSION:STRING=3.4.0 \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_ENABLE_PYTHON-INTERNAL:BOOL=ON \
-DVIAME_PYTHON_VERSION:STRING=3.6.15 \
-DVIAME_ENABLE_PYTORCH:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-INTERNAL:BOOL=ON \
-DVIAME_PYTORCH_VERSION:STRING=1.10.2 \
-DVIAME_ENABLE_PYTORCH-DISABLE-NINJA=ON \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=ON \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=ON \
-DVIAME_ENABLE_SEAL:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=ON \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=ON \
-DVIAME_ENABLE_VIVIA:BOOL=OFF \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_WEB_EXCLUDES:BOOL=ON \
-DVIAME_DOWNLOAD_MODELS:BOOL=ON \
-DVIAME_DOWNLOAD_MODELS-PYTORCH:BOOL=ON \
-DVIAME_DOWNLOAD_MODELS-ARCTIC-SEAL:BOOL=ON \
-DVIAME_DOWNLOAD_MODELS-HABCAM:BOOL=ON \
-DVIAME_DOWNLOAD_MODELS-MOUSS:BOOL=ON

# Perform multi-threaded build
make -j$(nproc) > build_log.txt 2>&1 || true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# HACK: Ensure invalid libsvm symlink isn't created
# Should be removed when this issue is fixed
rm install/lib/libsvm.so
cp install/lib/libsvm.so.2 install/lib/libsvm.so

# Remove all source files used for the build to save space and mode
# viame install to default linux install location
if [ -f "install/setup_viame.sh" ]; then
  cd /viame/build
  rm build_log.txt
  mkdir /opt/noaa
  mv install viame
  mv viame /opt/noaa
  cd /
  rm -rf /viame
  chown -R 1099:1099 /opt/noaa/viame
fi
