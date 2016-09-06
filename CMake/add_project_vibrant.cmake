# vibrant External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} vibrant )

ExternalProject_Add(vibrant
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/vibrant
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}

    # Required
    -Dvibrant_ENABLE_ARROWS:BOOL=ON
    -Dvibrant_ENABLE_TOOLS:BOOL=ON
    -Dvibrant_ENABLE_SPROKIT:BOOL=ON
    -Dvibrant_ENABLE_PROCESSES:BOOL=ON

    # Optional
    -Dvibrant_ENABLE_OPENCV:BOOL=${VIAME_ENABLE_OPENCV}
    -Dvibrant_ENABLE_VXL:BOOL=${VIAME_ENABLE_VXL}
    -Dvibrant_ENABLE_MATLAB:BOOL=${VIAME_ENABLE_MATLAB}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(vibrant forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/vibrant-stamp/vibrant-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_vibrant
  -Dvibrant_DIR:PATH=${VIAME_BUILD_PREFIX}/src/vibrant-build
  )
