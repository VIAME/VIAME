set(CTEST_SITE "zeah.kitware.com")
set(CTEST_BUILD_NAME "Windows7_GPU_Master_Nightly")
set(CTEST_SOURCE_DIRECTORY "C:/workspace/VIAME-Windows-GPU-MSI")
set(CTEST_BINARY_DIRECTORY "C:/workspace/VIAME-Windows-GPU-MSI/build/")
set(CTEST_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_PROJECT_NAME VIAME)
set(CTEST_BUILD_MODEL "Nightly")
set(CTEST_NIGHTLY_START_TIME "3:00:00 UTC")
set(CTEST_USE_LAUNCHERS 1)
include(CTestUseLaunchers)
set(OPTIONS 
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCUDA_NVCC_EXECUTABLE:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe"
  "-DCUDNN_ROOT_DIR:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
  "-DVIAME_FIXUP_BUNDLE=ON"
  "-DVIAME_ENABLE_CUDNN=ON"
  "-DVIAME_ENABLE_CUDA=ON"
  "-DVIAME_ENABLE_DARKNET=ON"
  "-DVIAME_ENABLE_DIVE=OFF"
  "-DVIAME_ENABLE_FFMPEG=ON"
  "-DVIAME_ENABLE_GDAL=OFF"
  "-DVIAME_ENABLE_LEARN=OFF"
  "-DVIAME_ENABLE_OPENCV=ON"
  "-DVIAME_OPENCV_VERSION=3.4.0"
  "-DVIAME_ENABLE_PYTHON=ON"
  "-DVIAME_ENABLE_PYTHON-INTERNAL=ON"
  "-DVIAME_ENABLE_SCALLOP_TK=OFF"
  "-DVIAME_ENABLE_SMQTK=ON"
  "-DVIAME_ENABLE_PYTORCH=OFF"
  "-DVIAME_ENABLE_VIVIA=OFF"
  "-DVIAME_DOWNLOAD_MODELS=ON"
  "-DVIAME_BUILD_KWIVER_DIR=C:/tmp/kv5"
  "-DVIAME_BUILD_PLUGINS_DIR=C:/tmp/vm5"
)

set(platform Windows7)
