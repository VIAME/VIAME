#! /bin/bash

# Debugging and logging
set -x
 
# Install Fletch & VIAME system deps
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
atlas-devel \
file \
which \
bzip2 \
bzip2-devel \
xz-devel

# Install x264 codec
yum -y install epel-release
wget https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
rpm -Uvh rpmfusion-free-release-7.noarch.rpm
yum -y install rpmfusion-free-release
yum -y install x264-devel

# Install and use more recent compiler
yum -y install centos-release-scl
yum -y install devtoolset-7
source /opt/rh/devtoolset-7/enable

# Install CMAKE
wget https://cmake.org/files/v3.17/cmake-3.17.0.tar.gz
tar zxvf cmake-3.*
cd cmake-3.17.0
./bootstrap --prefix=/usr/local --system-curl
make -j$(nproc)
make install
cd /
rm -rf cmake-3.17.0.tar.gz

# Update VIAME sub git sources
cd /viame/
git submodule update --init --recursive
mkdir build
cd build 

# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)
cp /viame/packages/patches/cuda/cuComplex.h /usr/local/cuda/include/

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$PATH:/viame/build/install/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/viame/build/install/lib:/viame/build/install/lib/python3.6
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/viame/build/install/include/python3.6m
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/viame/build/install/include/python3.6m

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CAFFE:BOOL=OFF \
-DVIAME_ENABLE_CAMTRAWL:BOOL=ON \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
-DVIAME_ENABLE_DIVE:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
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
make -j$(nproc)

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# HACK: Copy setup_viame.sh.install over setup_viame.sh
# Should be removed when this issue is fixed
cp ../cmake/setup_viame.sh.install install/setup_viame.sh

# HACK: Ensure invalid libsvm symlink isn't created
# Should be removed when this issue is fixed
rm install/lib/libsvm.so
cp install/lib/libsvm.so.2 install/lib/libsvm.so

# HACK: Copy in darknet executable
# Should be removed when this issue is fixed
cp build/src/darknet-build/darknet install/bin || true

# HACK: Copy in CUDA dlls missed by create_package
# Should be removed when this issue is fixed
cp -P /usr/local/cuda/lib64/libcudart.so* install/lib
cp -P /usr/local/cuda/lib64/libcusparse.so* install/lib
cp -P /usr/local/cuda/lib64/libcufft.so* install/lib
cp -P /usr/local/cuda/lib64/libcusolver.so* install/lib
cp -P /usr/local/cuda/lib64/libcublas.so* install/lib
cp -P /usr/local/cuda/lib64/libcublasLt.so* install/lib
cp -P /usr/local/cuda/lib64/libnvrtc* install/lib
cp -P /usr/local/cuda/lib64/libnvToolsExt.so* install/lib
cp -P /usr/local/cuda/lib64/libcurand.so* install/lib

cp -P /usr/lib64/libnccl.so* install/lib

# HACK: Copy in CUDNN missing .so files not included by
# create_package, should be removed when this issue is fixed
cp -P /usr/lib64/libcudnn.so.8* install/lib
cp -P /usr/lib64/libcudnn_adv_infer.so.8* install/lib
cp -P /usr/lib64/libcudnn_cnn_infer.so.8* install/lib
cp -P /usr/lib64/libcudnn_ops_infer.so.8* install/lib
cp -P /usr/lib64/libcudnn_cnn_train.so.8* install/lib
cp -P /usr/lib64/libcudnn_ops_train.so.8* install/lib
rm install/lib/libcudnn.so || true
rm install/lib/libcudnn_adv_infer.so || true
rm install/lib/libcudnn_cnn_infer.so || true
rm install/lib/libcudnn_ops_infer.so || true
rm install/lib/libcudnn_cnn_train.so || true
rm install/lib/libcudnn_ops_train.so || true
ln -s libcudnn.so.8 install/lib/libcudnn.so
ln -s libcudnn_adv_infer.so.8 install/lib/libcudnn_adv_infer.so
ln -s libcudnn_cnn_infer.so.8 install/lib/libcudnn_cnn_infer.so
ln -s libcudnn_ops_infer.so.8 install/lib/libcudnn_ops_infer.so
ln -s libcudnn_cnn_train.so.8 install/lib/libcudnn_cnn_train.so
ln -s libcudnn_ops_train.so.8 install/lib/libcudnn_ops_train.so

# HACK: Symlink CUDA so file link in pytorch directory for some
# systems with multiple CUDA 11s this is necessary
ln -s ../../../../libcublas.so.11 install/lib/python3.6/site-packages/torch/lib/libcublas.so.11

# HACK: Copy in other possible library requirements if present
# Should be removed when this issue is fixed
cp /usr/lib64/libva.so.1 install/lib || true
cp /usr/lib64/libreadline.so.6 install/lib || true
cp /usr/lib64/libdc1394.so.22 install/lib || true
cp /usr/lib64/libcrypto.so.10 install/lib || true
cp /usr/lib64/libpcre.so.1 install/lib || true
cp /usr/lib64/libgomp.so.1 install/lib || true
cp /usr/lib64/libSM.so.6 install/lib || true
cp /usr/lib64/libICE.so.6 install/lib || true
cp /usr/lib64/libblas.so.3 install/lib || true
cp /usr/lib64/liblapack.so.3 install/lib || true
cp /usr/lib64/libgfortran.so.3 install/lib || true
cp /usr/lib64/libquadmath.so.0 install/lib || true
cp /usr/lib64/libpng15.so.15 install/lib || true
cp /usr/lib64/libx264.so.148 install/lib || true
cp /usr/lib64/libx26410b.so.148 install/lib || true
