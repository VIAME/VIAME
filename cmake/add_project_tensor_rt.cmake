# TensorRT External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} tensor_rt )

if( VIAME_ENABLE_CUDA )
  FormatPassdowns( "CUDA" VIAME_TENSOR_RT_CUDA_FLAGS )
endif()

if( VIAME_ENABLE_CUDNN )
  FormatPassdowns( "CUDNN" VIAME_TENSOR_RT_CUDNN_FLAGS )
endif()

ExternalProject_Add( tensor_rt
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/tensor-rt
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_TENSOR_RT_CUDA_FLAGS}
    ${VIAME_TENSOR_RT_CUDNN_FLAGS}
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

if( VIAME_FORCEBUILD )
ExternalProject_Add_Step( tensor_rt forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/tensor-rt-stamp/tensor-rt-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
endif()

set( VIAME_ARGS_TENSOR_RT
  -DTENSOR_RT_DIR:PATH=${VIAME_BUILD_PREFIX}/src/tensor-rt-build
  )
