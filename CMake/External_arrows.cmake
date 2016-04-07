# arrows External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#
# Produced symbols are:
#   KWIVER_ARGS_arrows -
#

ExternalProject_Add(arrows_proj
  DEPENDS      vital_proj sprokit_proj
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/arrows
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
    -DARROWS_ENABLE_VXL:BOOL=ON
    -DARROWS_ENABLE_OPENCV:BOOL=${KWIVER_ENABLE_OPENCV}
    -DARROWS_ENABLE_DOCS:BOOL=${KWIVER_ENABLE_DOCS}
    -DARROWS_ENABLE_TESTS:BOOL=${KWIVER_ENABLE_TESTS}

  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(arrows_proj forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/arrows-stamp/arrows-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(KWIVER_ARGS_arrows
  -Darrows_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/lib/cmake
  )
