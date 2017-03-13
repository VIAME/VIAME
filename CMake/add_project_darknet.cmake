# darknet External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} darknet )

if( WIN32 )
  set( DARKNET_BUILD_SHARED OFF )
else()
  set( DARKNET_BUILD_SHARED ON )
endif()

if( VIAME_ENABLE_CUDA )
  FormatPassdowns( "CUDA" VIAME_CUDA_FLAGS )
endif()

ExternalProject_Add(darknet
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/darknet
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_CUDA}
    -DBUILD_SHARED_LIBS:BOOL=${DARKNET_BUILD_SHARED}
    -DINSTALL_HEADER_FILES:BOOL=ON
    -DUSE_GPU:BOOL=${VIAME_ENABLE_CUDA}
    -DUSE_CUDNN:BOOL=${VIAME_ENABLE_CUDNN}
    -DUSE_OPENCV:BOOL=${VIAME_ENABLE_OPENCV}
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(darknet forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/darknet-stamp/darknet-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_darknet
  -Ddarknet_DIR:PATH=${VIAME_BUILD_PREFIX}/src/darknet-build
  )
