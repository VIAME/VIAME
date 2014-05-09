# map_tk External Project
#
# Required symbols are:
#   KWIVER_BUILD_PREFIX - where packages are built
#   KWIVER_BUILD_INSTALL_PREFIX - directory install target
#   KWIVER_PACKAGES_DIR - location of git submodule packages
#   KWIVER_ARGS_COMMON -
#
# Produced symbols are:
#   KWIVER_ARGS_map_tk -
#

ExternalProject_Add(map_tk
  PREFIX ${KWIVER_BUILD_PREFIX}
  SOURCE_DIR ${KWIVER_PACKAGES_DIR}/map_tk
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${KWIVER_ARGS_COMMON}

  INSTALL_DIR ${KWIVER_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(map_tk forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${KWIVER_BUILD_PREFIX}/src/map_tk-stamp/map_tk-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set(KWIVER_ARGS_map_tk
  -Dmaptk_DIR:PATH=${KWIVER_BUILD_INSTALL_PREFIX}/lib/cmake
  )
