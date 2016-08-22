# maptk External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
#

ExternalProject_Add(kwiver
  DEPENDS fletch
  PREFIX ${CMAKE_BINARY_DIR}/build
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/kwiver
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    -DVIAME_ENABLE_DOCS:BOOL=${VIAME_ENABLE_DOCS}
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(kwiver forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/kwiver-stamp/kwiver-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(VIAME_ARGS_kwiver
  -Dkwiver_DIR:PATH=${VIAME_BUILD_INSTALL_PREFIX}/lib/cmake
  )
