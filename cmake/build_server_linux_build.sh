#! /bin/bash

# Perform multi-threaded build
make -j$(nproc)

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# Get paths to system libraries if not entered as arguments
if [ -n "$1" ]; then
  export CUDA_BASE=$1
else
  export CUDA_BASE=/usr/local/cuda
fi

if [ -n "$2" ]; then
  export CUDNN_BASE=$2
elif cat /etc/os-release | grep 'Ubuntu'; then
  export CUDNN_BASE=/usr/lib/x86_64-linux-gnu
else
  export CUDNN_BASE=/usr/lib64
fi

if [ -n "$3" ]; then
  export LIB_BASE=$3
elif cat /etc/os-release | grep 'Ubuntu'; then
  export LIB_BASE=/usr/lib/x86_64-linux-gnu
else
  export LIB_BASE=/usr/lib64
fi

# HACK: Ensure invalid libsvm symlink isn't created
# Should be removed when this issue is fixed
rm install/lib/libsvm.so
cp install/lib/libsvm.so.2 install/lib/libsvm.so

# HACK: Ensure all cuda libraries 
# Should be removed when this issue is fixed
if [ -d "${CUDA_BASE}" ]; then
  cp -P ${CUDA_BASE}/lib64/libcudart.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcusparse.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcufft.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcusolver.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcublas.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcublasLt.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcupti.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libcurand.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libnvjpeg.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libnvJitLink.so* install/lib
  cp -P ${CUDA_BASE}/lib64/libnvrtc* install/lib
  cp -P ${CUDA_BASE}/lib64/libnvToolsExt.so* install/lib

  cp -P ${CUDA_BASE}/targets/x86_64-linux/lib/libcupti.so* install/lib
  cp -P ${CUDA_BASE}/targets/x86_64-linux/lib/libcufile.so* install/lib

  if [ is_ubuntu ]; then
    cp -P ${CUDA_BASE}/lib64/libnppi* install/lib
    cp -P ${CUDA_BASE}/lib64/libnppc* install/lib
  fi

  # HACK: Symlink CUDA library in pytorch directory
  # For some systems with multiple CUDA 11s installed this is necessary
  export TORCH_BASE=install/lib/python3.10/site-packages/torch
  if [ -d "${TORCH_BASE}" ]; then
    ln -s ../../../../libcublas.so.12 ${TORCH_BASE}/lib/libcublas.so.12
  fi
fi

# HACK: Copy in CUDNN missing .so files not included by
# create_package, should be removed when this issue is fixed
if [ -d "${CUDA_BASE}" ]; then
  cp -P ${CUDNN_BASE}/libcudnn.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_adv.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_cnn.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_ops.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_engines_precompiled.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_engines_runtime_compiled.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_graph.so.9* install/lib
  cp -P ${CUDNN_BASE}/libcudnn_heuristic.so.9* install/lib

  rm install/lib/libcudnn.so || true
  rm install/lib/libcudnn_adv.so || true
  rm install/lib/libcudnn_cnn.so || true
  rm install/lib/libcudnn_ops.so || true
  rm install/lib/libcudnn_engines_precompiled.so || true
  rm install/lib/libcudnn_engines_runtime_compiled.so || true
  rm install/lib/libcudnn_graph.so || true
  rm install/lib/libcudnn_heuristic.so || true

  ln -s libcudnn.so.9 install/lib/libcudnn.so
  ln -s libcudnn_adv.so.9 install/lib/libcudnn_adv.so
  ln -s libcudnn_cnn.so.9 install/lib/libcudnn_cnn.so
  ln -s libcudnn_ops.so.9 install/lib/libcudnn_ops.so
  ln -s libcudnn_engines_precompiled.so.9 install/lib/libcudnn_engines_precompiled.so
  ln -s libcudnn_engines_runtime_compiled.so.9 install/lib/libcudnn_engines_runtime_compiled.so
  ln -s libcudnn_graph.so.9 install/lib/libcudnn_graph.so
  ln -s libcudnn_heuristic.so.9 install/lib/libcudnn_heuristic.so
fi

# HACK: Copy in other possible library requirements if present
# Should be removed when this issue is fixed
cp ${LIB_BASE}/libcrypt.so.2 install/lib || true
cp ${LIB_BASE}/libcrypt.so.2.0.0 install/lib || true
cp ${LIB_BASE}/libffi.so.6 install/lib || true
cp ${LIB_BASE}/libva.so.1 install/lib || true
cp ${LIB_BASE}/libssl.so.10 install/lib || true
cp ${LIB_BASE}/libreadline.so.6 install/lib || true
cp ${LIB_BASE}/libdc1394.so.22 install/lib || true
cp ${LIB_BASE}/libcrypto.so.10 install/lib || true
cp ${LIB_BASE}/libpcre.so.1 install/lib || true
cp ${LIB_BASE}/libgomp.so.1 install/lib || true
cp ${LIB_BASE}/libSM.so.6 install/lib || true
cp ${LIB_BASE}/libICE.so.6 install/lib || true
cp ${LIB_BASE}/libblas.so.3 install/lib || true
cp ${LIB_BASE}/liblapack.so.3 install/lib || true
cp ${LIB_BASE}/libgfortran.so.4 install/lib || true
cp ${LIB_BASE}/libgfortran.so.5 install/lib || true
cp ${LIB_BASE}/libquadmath.so.0 install/lib || true
cp ${LIB_BASE}/libpng15.so.15 install/lib || true
cp ${LIB_BASE}/libxcb.so.1 install/lib || true
cp ${LIB_BASE}/libXau.so.6 install/lib || true

# HACK: Copy in other possible library requirements if present
# Should be removed when this issue is fixed
if cat /etc/os-release | grep 'Ubuntu'; then
  cp /lib/x86_64-linux-gnu/libreadline.so.6 install/lib || true
  cp /lib/x86_64-linux-gnu/libreadline.so.7 install/lib || true
  cp /lib/x86_64-linux-gnu/libpcre.so.3 install/lib || true
  cp /lib/x86_64-linux-gnu/libexpat.so.1 install/lib || true

  cp ${LIB_BASE}/libcrypto.so install/lib || true
  cp ${LIB_BASE}/libcrypto.so.1.1 install/lib || true
  cp ${LIB_BASE}/libfreetype.so.6 install/lib || true
  cp ${LIB_BASE}/libharfbuzz.so.0 install/lib || true
  cp ${LIB_BASE}/libpng16.so.16 install/lib || true
  cp ${LIB_BASE}/libglib-2.0.so.0 install/lib || true
  cp ${LIB_BASE}/libgraphite2.so.3 install/lib || true
fi
