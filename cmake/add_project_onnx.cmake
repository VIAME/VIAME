# Install the onnxruntime C++ runtime libs. The matching Python package
# (onnxruntime / onnxruntime-gpu, version 1.23.2) is installed alongside
# the rest of the pinned wheels by add_project_python_deps; that pip
# install is what `python-deps` ultimately provides, so we depend on it
# here to ordering: the libs are unpacked into the same site-packages
# subtree (${VIAME_PYTHON_PACKAGES}/onnxruntime/onnxruntimelibs) and
# would race with the wheel install if run earlier.
#
# Note the version mismatch is intentional: the prebuilt C++ libs come
# from upstream's 1.12.1 release archive (a stable headers/.so set used
# by mmdetection's ONNX export tooling), while the Python wheel tracks
# the newer 1.23.x line for CUDA 12 inference.
set( ONNXRUNTIME_LIB_URL "" )
set( ONNXRUNTIME_LIB_DOWNLOAD_DIR ${VIAME_BUILD_PREFIX}/src/onnxruntimelibs )
set( ONNXRUNTIME_LIB_INSTALL_DIR ${VIAME_PYTHON_PACKAGES}/onnxruntime )

if( UNIX )
  set( ONNXRUNTIME_LIB_URL https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz )
elseif( WIN32 )
  set( ONNXRUNTIME_LIB_URL https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-win-x64-1.12.1.zip )
else()
  message( FATAL_ERROR "Cannot download ONNXRUNTIME because system is not supported" )
endif()


set( ONNXRUNTIME_LIB_DEPS python-deps )
set( ONNXRUNTIME_LIB_INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${ONNXRUNTIME_LIB_DOWNLOAD_DIR} ${ONNXRUNTIME_LIB_INSTALL_DIR}/onnxruntimelibs )

ExternalProject_Add( onnxruntimelibs
  DEPENDS ${ONNXRUNTIME_LIB_DEPS}
  SOURCE_DIR ""
  URL ${ONNXRUNTIME_LIB_URL}
  PREFIX ${VIAME_BUILD_PREFIX}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${ONNXRUNTIME_LIB_INSTALL_COMMAND}
  )
