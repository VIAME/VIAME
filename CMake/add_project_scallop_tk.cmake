# scallop_tk External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} scallop_tk )

ExternalProject_Add(scallop_tk
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/scallop-tk
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DVC_TOOLNAMES:BOOL=ON
    -DBUILD_TOOLS:BOOL=ON
    -DBUILD_TESTS:BOOL=OFF
    -DENABLE_CAFFE:BOOL=${VIAME_ENABLE_CAFFE}
    -DCAFFE_CPU_ONLY:BOOL=${VIAME_DISABLE_GPU_SUPPORT}
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(scallop_tk forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/scallop_tk-stamp/scallop_tk-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_scallop_tk
  -DScallopTK_DIR:PATH=${VIAME_BUILD_PREFIX}/src/scallop_tk-build
  )
