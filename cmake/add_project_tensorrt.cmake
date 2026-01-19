# TensorRT External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} tensorrt )

option( TENSORRT_LIB_DIR "TensorRT Non-Public Binary Path" "" )

if( NOT TENSORRT_LIB_DIR )
  message( FATAL_ERROR "TensorRT LIB DIR must be set.\n"
    "Download from https://github.com/NVIDIA/TensorRT.\n" )
endif()

if( VIAME_ENABLE_CUDA )
  FormatPassdowns( "CUDA" VIAME_TENSORRT_CUDA_FLAGS )
endif()

if( VIAME_ENABLE_CUDNN )
  FormatPassdowns( "CUDNN" VIAME_TENSORRT_CUDNN_FLAGS )
endif()

ExternalProject_Add( tensorrt
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/tensorrt
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_TENSORRT_CUDA_FLAGS}
    ${VIAME_TENSORRT_CUDNN_FLAGS}
    -DTRT_LIB_DIR:PATH=${TENSORRT_LIB_DIR}
    -DTRT_OUT_DIR:PATH=${VIAME_BUILD_INSTALL_PREFIX}
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

if( VIAME_BUILD_FORCE_REBUILD )
  RemoveProjectCMakeStamp( tensorrt )
endif()

set( VIAME_ARGS_TENSOR_RT
  -DTENSORRT_DIR:PATH=${VIAME_BUILD_PREFIX}/src/tensorrt-build
  )
