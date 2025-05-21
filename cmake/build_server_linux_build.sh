#! /bin/bash

# Perform multi-threaded build
make -j$(nproc)

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# HACK: Ensure invalid libsvm symlink isn't created
# Should be removed when this issue is fixed
rm install/lib/libsvm.so
cp install/lib/libsvm.so.2 install/lib/libsvm.so

# HACK: Copy in CUDA dlls missed by create_package
# Should be removed when this issue is fixed
if cat /etc/os-release | grep 'Ubuntu'; then
  export LIBBASE=/usr/lib/x86_64-linux-gnu
  export CUDABASE=/usr/local/cuda
else
  export LIBBASE=/usr/lib64
  export CUDABASE=/usr/local/cuda
fi

if [ ! -d $CUDABASE ]; then
  export CUDABASE=/usr/local/cuda-viame
fi

if [ -d "${CUDABASE}" ]; then
  cp -P ${CUDABASE}/lib64/libcudart.so* install/lib
  cp -P ${CUDABASE}/lib64/libcusparse.so* install/lib
  cp -P ${CUDABASE}/lib64/libcufft.so* install/lib
  cp -P ${CUDABASE}/lib64/libcusolver.so* install/lib
  cp -P ${CUDABASE}/lib64/libcublas.so* install/lib
  cp -P ${CUDABASE}/lib64/libcublasLt.so* install/lib
  cp -P ${CUDABASE}/lib64/libcupti.so* install/lib
  cp -P ${CUDABASE}/lib64/libcurand.so* install/lib
  cp -P ${CUDABASE}/lib64/libnvjpeg.so* install/lib
  cp -P ${CUDABASE}/lib64/libnvJitLink.so* install/lib
  cp -P ${CUDABASE}/lib64/libnvrtc* install/lib
  cp -P ${CUDABASE}/lib64/libnvToolsExt.so* install/lib

  cp -P ${CUDABASE}/targets/x86_64-linux/lib/libcupti.so* install/lib
  cp -P ${CUDABASE}/targets/x86_64-linux/lib/libcufile.so* install/lib

  if [ is_ubuntu ]; then
    cp -P ${CUDABASE}/lib64/libnppi* install/lib
    cp -P ${CUDABASE}/lib64/libnppc* install/lib
  fi

  # HACK: Symlink CUDA library in pytorch directory
  # For some systems with multiple CUDA 11s installed this is necessary
  export TORCHBASE=install/lib/python3.10/site-packages/torch
  if [ -d "${TORCHBASE}" ]; then
    ln -s ../../../../libcublas.so.12 ${TORCHBASE}/lib/libcublas.so.12
  fi
fi

# HACK: Copy in CUDNN missing .so files not included by
# create_package, should be removed when this issue is fixed
if [ -d "${CUDABASE}" ]; then
  cp -P ${LIBBASE}/libcudnn.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_adv.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_cnn.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_ops.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_engines_precompiled.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_engines_runtime_compiled.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_graph.so.9* install/lib
  cp -P ${LIBBASE}/libcudnn_heuristic.so.9* install/lib

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
cp ${LIBBASE}/libcrypt.so.2 install/lib || true
cp ${LIBBASE}/libcrypt.so.2.0.0 install/lib || true
cp ${LIBBASE}/libffi.so.6 install/lib || true
cp ${LIBBASE}/libva.so.1 install/lib || true
cp ${LIBBASE}/libssl.so.10 install/lib || true
cp ${LIBBASE}/libreadline.so.6 install/lib || true
cp ${LIBBASE}/libdc1394.so.22 install/lib || true
cp ${LIBBASE}/libcrypto.so.10 install/lib || true
cp ${LIBBASE}/libpcre.so.1 install/lib || true
cp ${LIBBASE}/libgomp.so.1 install/lib || true
cp ${LIBBASE}/libSM.so.6 install/lib || true
cp ${LIBBASE}/libICE.so.6 install/lib || true
cp ${LIBBASE}/libblas.so.3 install/lib || true
cp ${LIBBASE}/liblapack.so.3 install/lib || true
cp ${LIBBASE}/libgfortran.so.4 install/lib || true
cp ${LIBBASE}/libgfortran.so.5 install/lib || true
cp ${LIBBASE}/libquadmath.so.0 install/lib || true
cp ${LIBBASE}/libpng15.so.15 install/lib || true
cp ${LIBBASE}/libxcb.so.1 install/lib || true
cp ${LIBBASE}/libXau.so.6 install/lib || true

# HACK: Copy in other possible library requirements if present
# Should be removed when this issue is fixed
if cat /etc/os-release | grep 'Ubuntu'; then
  cp /lib/x86_64-linux-gnu/libreadline.so.6 install/lib || true
  cp /lib/x86_64-linux-gnu/libreadline.so.7 install/lib || true
  cp /lib/x86_64-linux-gnu/libpcre.so.3 install/lib || true
  cp /lib/x86_64-linux-gnu/libexpat.so.1 install/lib || true

  cp ${LIBBASE}/libcrypto.so install/lib || true
  cp ${LIBBASE}/libcrypto.so.1.1 install/lib || true
  cp ${LIBBASE}/libfreetype.so.6 install/lib || true
  cp ${LIBBASE}/libharfbuzz.so.0 install/lib || true
  cp ${LIBBASE}/libpng16.so.16 install/lib || true
  cp ${LIBBASE}/libglib-2.0.so.0 install/lib || true
  cp ${LIBBASE}/libgraphite2.so.3 install/lib || true
fi
