# Seal-TK External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} seal_tk )

ExternalProject_Add(seal_tk
  DEPENDS fletch kwiver
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/seal-tk
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}
    ${VIAME_ARGS_Qt}
    -DQt5_DIR:PATH=${VIAME_BUILD_INSTALL_PREFIX}/lib/cmake/Qt5
    -DBUILD_SHARED_LIBS:BOOL=ON

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

if( VIAME_FORCEBUILD )
  ExternalProject_Add_Step(seal_tk forcebuild
    COMMAND ${CMAKE_COMMAND}
      -E remove ${VIAME_BUILD_PREFIX}/src/seal_tk-stamp/seal_tk-build
    COMMENT "Removing build stamp file for build update (forcebuild)."
    DEPENDEES configure
    DEPENDERS build
    ALWAYS 1
    )
endif()

set(VIAME_ARGS_seal_tk
  -Dseal_tk_DIR:PATH=${VIAME_BUILD_PREFIX}/src/seal_tk-build
  )

