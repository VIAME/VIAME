# sprokit External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#
# Produced symbols are:
#   KWIVER_ARGS_sprokit -
#

ExternalProject_Add(sprokit
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/sprokit
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
    -DSPROKIT_ENABLE_PYTHON:BOOL=YES

  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(sprokit forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/sprokit-stamp/sprokit-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(KWIVER_ARGS_sprokit
  -Dsprokit_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/lib/cmake
  )
