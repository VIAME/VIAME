# VIBRANT External Project
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


ExternalProject_Add(vibrant
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/vibrant
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  DEPENDS VXL
  )

ExternalProject_Add_Step(vibrant forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/vibrant-stamp/vibrant-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

#include_directories( ${KWIVER_BUILD_INSTALL_PREFIX}/include/vxl
#                     ${KWIVER_BUILD_INSTALL_PREFIX}/include/vxl/vcl
#                     ${KWIVER_BUILD_INSTALL_PREFIX}/include/vxl/core )
#
#set(KWIVER_ARGS_VXL
#  #-DVXL_DIR=${KWIVER_BUILD_PREFIX}/src/VXL-build
#  -DVXL_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/share/vxl/cmake
#  )
