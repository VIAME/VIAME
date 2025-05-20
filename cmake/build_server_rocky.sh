#! /bin/bash

# Debugging, logging, and options
set -x

export VIAME_SOURCE_DIR=/viame
export VIAME_BUILD_DIR=$VIAME_SOURCE_DIR/build
export VIAME_INSTALL_DIR=$VIAME_BUILD_DIR/install

# Install system dependencies and use more recent compiler
$VIAME_SOURCE_DIR/cmake/build_server_deps_yum.sh

# Install more modern CMAKE and OpenSSL from source
./viame/cmake/build_server_linux_ssl.sh
./viame/cmake/build_server_linux_cmake.sh

# Hack for certain versions of cudnn installs on some OS
if [ -f /usr/include/cudnn_v9.h ] && [ ! -f /usr/include/cudnn.h ]; then
 ln -s /usr/include/cudnn_v9.h /usr/include/cudnn.h
 ln -s /usr/include/cudnn_adv_v9.h /usr/include/cudnn_adv.h
 ln -s /usr/include/cudnn_cnn_v9.h/usr/include/cudnn_cnn.h
 ln -s /usr/include/cudnn_ops_v9.h /usr/include/cudnn_ops.h
 ln -s /usr/include/cudnn_version_v9.h /usr/include/cudnn_version.h
 ln -s /usr/include/cudnn_backend_v9.h /usr/include/cudnn_backend.h
 ln -s /usr/include/cudnn_graph_v9.h /usr/include/cudnn_graph.h
fi

# Update VIAME sub git sources
echo "Checking out VIAME submodules"

cd $VIAME_SOURCE_DIR
git config --global --add safe.directory $VIAME_SOURCE_DIR
git submodule update --init --recursive
mkdir build
cd build

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$VIAME_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$VIAME_INSTALL_DIR/lib:$VIAME_INSTALL_DIR/lib/python3.10:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$CPLUS_INCLUDE_PATH

# Configure VIAME
echo "Beginning VIAME CMake configuration"

cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
-DVIAME_ENABLE_DIVE:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
-DVIAME_ENABLE_GDAL:BOOL=OFF \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=OFF \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_LEARN:BOOL=ON \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_ONNX:BOOL=ON \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_OPENCV_VERSION:STRING=3.4.0 \
-DVIAME_ENABLE_POSTGRESQL=OFF \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_PYTHON_BUILD_FROM_SOURCE:BOOL=ON \
-DVIAME_PYTHON_VERSION:STRING=3.10.4 \
-DVIAME_ENABLE_PYTORCH:BOOL=ON \
-DVIAME_PYTORCH_BUILD_FROM_SOURCE:BOOL=ON \
-DVIAME_PYTORCH_DISABLE_NINJA=OFF \
-DVIAME_PYTORCH_VERSION:STRING=2.7.0 \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-SAM:BOOL=ON \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=OFF \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_DARKNET:BOOL=ON 

# Prevent download from notoriously bad opencv repo
curl https://data.kitware.com/api/v1/item/682bee3b0dcd2dfb445a5401/download --output tmp.tar.gz
tar -xvf tmp.tar.gz
rm tmp.tar.gz

# Build VIAME, pipe output to file
echo "Beginning core build, routing build info to build_log.txt"

../cmake/build_server_linux_build.sh > build_log.txt 2>&1

# Output check statments
if grep -q "Built target viame" build_log.txt; then
  echo "VIAME Build Succeeded"

  # Make zip file of install
  mv install viame
  rm VIAME-v1.0.0-Linux-64Bit.tar.gz ||:
  tar -zcvf VIAME-v1.0.0-Linux-64Bit.tar.gz viame
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
