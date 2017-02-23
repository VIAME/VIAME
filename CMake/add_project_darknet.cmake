# darknet External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} darknet )

if( ${VIAME_DISABLE_GPU_SUPPORT} )
  set( DARKNET_USE_GPU OFF )
else()
  set( DARKNET_USE_GPU ON )
endif()

if( ${VIAME_DISABLE_CUDNN_SUPPORT} )
  set( DARKNET_USE_CUDNN OFF )
else()
  set( DARKNET_USE_CUDNN OFF )
endif()

ExternalProject_Add(darknet
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/darknet
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    -DINSTALL_HEADER_FILES:BOOL=ON
    -DUSE_GPU:BOOL=${DARKNET_USE_GPU}
    -DUSE_CUDNN:BOOL=${DARKNET_USE_CUDNN}
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
