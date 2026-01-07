# Install onnx python
set( ONNXRUNTIME_PYTHON onnxruntime )
set( ONNXRUNTIME_VERSION 1.12.1 )
set( ONNXRUNTIME_DOWNLOAD_DIR ${VIAME_PACKAGES_DIR}/downloads )

set( ONNXRUNTIME_PIP_INSTALL_CMD
  ${CMAKE_COMMAND}
    -DPYTHON_EXECUTABLE=${Python_EXECUTABLE}
    -DPython_EXECUTABLE=${Python_EXECUTABLE}
    -DWHEEL_DIR=${ONNXRUNTIME_DOWNLOAD_DIR}
    -P ${VIAME_CMAKE_DIR}/install_python_wheel.cmake )

set( ONNXRUNTIME_PYTHON_DOWNLOAD ${Python_EXECUTABLE} -m pip download
  --no-deps onnxruntime==${ONNXRUNTIME_VERSION} -d "${ONNXRUNTIME_DOWNLOAD_DIR}" )

# Convert install command and env vars to ----separated strings for the wrapper script
string( REPLACE ";" "----" ONNX_INSTALL_CMD_STR "${ONNXRUNTIME_PIP_INSTALL_CMD}" )
string( REPLACE ";" "----" ONNX_ENV_STR "${PYTHON_DEP_ENV_VARS}" )

set( ONNXRUNTIME_PYTHON_INSTALL
  ${CMAKE_COMMAND}
    -DCOMMAND_TO_RUN=${ONNX_INSTALL_CMD_STR}
    -DENV_VARS=${ONNX_ENV_STR}
    -P ${VIAME_CMAKE_DIR}/run_python_command.cmake )

ExternalProject_Add( ${ONNXRUNTIME_PYTHON}
  DEPENDS fletch python-deps
  PREFIX ${VIAME_BUILD_PREFIX}
  DOWNLOAD_COMMAND ${ONNXRUNTIME_PYTHON_DOWNLOAD}
  SOURCE_DIR ""
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${ONNXRUNTIME_PYTHON_INSTALL}
  LIST_SEPARATOR "----"
  )

# Install onnx libs
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


set( ONNXRUNTIME_LIB_DEPS ${ONNXRUNTIME_PYTHON} )
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
