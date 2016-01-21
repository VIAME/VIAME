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


ExternalProject_Add(kwant
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/kwant
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  DEPENDS VXL
  )

ExternalProject_Add_Step(kwant forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/kwant-stamp/kwant-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

