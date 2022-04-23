
# Perform multi-threaded build
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
if [ -d "/usr/local/cuda" ]; then
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

  # HACK: Symlink CUDA so file link in pytorch directory for some
  # systems with multiple CUDA 11s this is necessary
  ln -s ../../../../libcublas.so.11 install/lib/python3.8/site-packages/torch/lib/libcublas.so.11
fi

# HACK: Copy in CUDNN missing .so files not included by
# create_package, should be removed when this issue is fixed
if [ -d "/usr/local/cuda" ]; then
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
fi

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
