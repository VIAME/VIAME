# darknet External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} darknet )

ExternalProject_Add(darknet
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/darknet
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${ScallopTK_BUILD_SHARED}
    -DVC_TOOLNAMES:BOOL=ON
    -DBUILD_TOOLS:BOOL=ON
    -DBUILD_TESTS:BOOL=OFF
    -DENABLE_CAFFE:BOOL=${VIAME_ENABLE_CAFFE}
    -DCAFFE_CPU_ONLY:BOOL=${VIAME_DISABLE_GPU_SUPPORT}
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
