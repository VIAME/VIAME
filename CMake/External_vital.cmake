# maptk External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#
# Produced symbols are:
#   KWIVER_ARGS_vatal -
#

ExternalProject_Add(vital_proj
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/vital
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}
    -DKWIVER_ENABLE_DOCS:BOOL=${KWIVER_ENABLE_DOCS}
    -DVITAL_ENABLE_LOG4CXX:BOOL=${KWIVER_ENABLE_LOG4CXX}
    -DVITAL_ENABLE_C_LIB:BOOL=${KWIVER_ENABLE_PYTHON}
    -DVITAL_ENABLE_PYTHON:BOOL=${KWIVER_ENABLE_PYTHON}
  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(vital_proj forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/vital-stamp/vital-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

include_directories( ${KWIVER_BUILD_INSTALL_PREFIX}/include/kwiver)

set(KWIVER_ARGS_vital
  -Dvital_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/lib/cmake
  )
