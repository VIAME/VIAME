# fletch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} fletch )

if( WIN32 AND VIAME_ENABLE_VXL )
  set( fletch_VXL_DEP_FLAGS
    -Dfletch_ENABLE_ZLib:BOOL=${VIAME_ENABLE_VXL}
    -Dfletch_ENABLE_libjpeg-turbo:BOOL=${VIAME_ENABLE_VXL}
    -Dfletch_ENABLE_libtiff:BOOL=${VIAME_ENABLE_VXL}
  )
endif()

ExternalProject_Add(fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/fletch
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}

    # KWIVER Dependencies, Always On
    -Dfletch_ENABLE_Boost:BOOL=TRUE
    -Dfletch_ENABLE_Eigen:BOOL=TRUE

    # System Related Options
    -Dfletch_DISABLE_GPU_SUPPORT:BOOL=${VIAME_DISABLE_GPU_SUPPORT}
    -Dfletch_DISABLE_FFMPEG_SUPPORT:BOOL=${VIAME_DISABLE_FFMPEG_SUPPORT}

    # Optional Dependencies
    -Dfletch_ENABLE_VXL:BOOL=${VIAME_ENABLE_VXL}
    ${fletch_VXL_DEP_FLAGS}

    -Dfletch_ENABLE_OpenCV:BOOL=${VIAME_ENABLE_OPENCV}

    -Dfletch_ENABLE_Caffe:BOOL=${VIAME_ENABLE_CAFFE}
    -DAUTO_ENABLE_CAFFE_DEPENDENCY:BOOL=${VIAME_ENABLE_CAFFE}

    # Set fletch install path to be viame install path
    -Dfletch_BUILD_INSTALL_PREFIX:PATH=${VIAME_BUILD_INSTALL_PREFIX}

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

if( VIAME_ENABLE_OPENCV )
  set(VIAME_ARGS_fletch
    ${VIAME_ARGS_fletch}
    -DOpenCV_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/OpenCV-build
    )
endif()

if( VIAME_ENABLE_CAFFE )
  set(VIAME_ARGS_fletch
     ${VIAME_ARGS_fletch}
    -DCaffe_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/Caffe-build
    )
endif()
