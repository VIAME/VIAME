# maptk External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#
# Produced symbols are:
#   KWIVER_ARGS_maptk -
#

ExternalProject_Add(maptk
  DEPENDS VXL
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/maptk
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
    -DMAPTK_ENABLE_VXL:BOOL=ON
  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(maptk forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/maptk-stamp/maptk-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(KWIVER_ARGS_maptk
  -Dmaptk_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/lib/cmake
  )
