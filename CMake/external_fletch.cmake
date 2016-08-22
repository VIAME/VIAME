# maptk External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

if( ${VIAME_ENABLE_CAFFE} OR ${VIAME_ENABLE_SCALLOP_TK} )
  set( VIAME_ENABLE_OPENCV TRUE )
endif()

ExternalProject_Add(fletch
  PREFIX ${CMAKE_BINARY_DIR}/build
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/fletch
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}

    # KWIVER Dependencies, Always on
    -Dfletch_ENABLE_Boost:BOOL=TRUE
    -Dfletch_ENABLE_Eigen:BOOL=TRUE

    # Optional Dependencies
    -Dfletch_ENABLE_VXL:BOOL=${VIAME_ENABLE_VXL}
    -Dfletch_ENABLE_OpenCV:BOOL=${VIAME_ENABLE_OPENCV}
    -Dfletch_ENABLE_Caffe:BOOL=${VIAME_ENABLE_CAFFE}
    -DAUTO_ENABLE_CAFFE_DEPENDENCY:BOOL=${VIAME_ENABLE_CAFFE}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  INSTALL_COMMAND ""
  )

ExternalProject_Add_Step(fletch forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/fletch-stamp/fletch-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_fletch
  -Dfletch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build
  )
