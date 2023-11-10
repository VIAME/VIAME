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
libffi-devel \
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
xz-devel \
vim

# Install and use more recent compiler
yum -y install centos-release-scl
yum -y install devtoolset-7
source /opt/rh/devtoolset-7/enable

# Install CMAKE
wget https://cmake.org/files/v3.23/cmake-3.23.1.tar.gz
tar zxvf cmake-3.*
cd cmake-3.23.1
./bootstrap --prefix=/usr/local --system-curl
make -j$(nproc)
make install
cd /
rm -rf cmake-3.23.1.tar.gz

# Install NINJA for faster builds of some dependencies
rpm -ivh https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/n/ninja-build-1.10.2-3.el7.x86_64.rpm

# Update VIAME sub git sources
cd /viame/
git submodule update --init --recursive
mkdir build
cd build 

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=/viame/build/install/bin:$PATH
export LD_LIBRARY_PATH=/viame/build/install/lib:/viame/build/install/lib/python3.6:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/viame/build/install/include/python3.6m:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/viame/build/install/include/python3.6m:$CPLUS_INCLUDE_PATH

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CAFFE:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=OFF \
-DVIAME_ENABLE_CUDNN:BOOL=OFF \
-DVIAME_ENABLE_DIVE:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
-DVIAME_ENABLE_GDAL:BOOL=ON \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=OFF \
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
-DVIAME_ENABLE_PYTORCH-DISABLE-NINJA=ON \
-DVIAME_PYTORCH_VERSION:STRING=1.10.2 \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=OFF \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_DARKNET:BOOL=ON 

# Build VIAME, pipe output to file
../cmake/linux_release_build.sh > build_log.txt 2>&1

# Rename output
mv VIAME.tar.gz VIAME-CPU-v1.0.0-Linux-64Bit.tar.gz

# Output check statments
if grep -q "Built target viame" build_log.txt; then
  echo "VIAME Build Succeeded"

  # Make zip file of install
  mv install viame
  rm VIAME-CPU-v1.0.0-Linux-64Bit.tar.gz ||:
  tar -zcvf VIAME.tar.gz viame
  mv viame install
else
  echo "VIAME Build Failed"
  exit 1
fi

if  grep -q "fixup_bundle: preparing..." build_log.txt; then
  echo "Fixup Bundle Called Succesfully"
else
  echo "Error: Fixup Bundle Not Called"
  exit 1
fi
