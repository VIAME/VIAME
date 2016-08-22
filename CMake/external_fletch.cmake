# maptk External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
#

ExternalProject_Add(fletch
  PREFIX ${CMAKE_BINARY_DIR}/build
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/fletch
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    -DVIAME_ENABLE_DOCS:BOOL=${VIAME_ENABLE_DOCS}
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
  -Dfletch_DIR:PATH=${VIAME_BUILD_INSTALL_PREFIX}/lib/cmake
  )
