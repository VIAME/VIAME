# vivia External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} vivia )

ExternalProject_Add(vivia
  DEPENDS fletch vibrant
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/vivia
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}

    # Required
    -Dvivia_ENABLE_ARROWS:BOOL=ON
    -Dvivia_ENABLE_TOOLS:BOOL=ON
    -Dvivia_ENABLE_SPROKIT:BOOL=ON
    -Dvivia_ENABLE_PROCESSES:BOOL=ON

    # Optional
    -Dvivia_ENABLE_OPENCV:BOOL=${VIAME_ENABLE_OPENCV}
    -Dvivia_ENABLE_VXL:BOOL=${VIAME_ENABLE_VXL}
    -Dvivia_ENABLE_MATLAB:BOOL=${VIAME_ENABLE_MATLAB}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(vivia forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/vivia-stamp/vivia-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_vivia
  -Dvivia_DIR:PATH=${VIAME_BUILD_PREFIX}/src/vivia-build
  )
