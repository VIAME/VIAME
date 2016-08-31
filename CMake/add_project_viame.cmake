# viame External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

FormatPassdowns( "VIAME_ENABLE" VIAME_ENABLE_FLAGS )
FormatPassdowns( "VIAME_DISABLE" VIAME_DISABLE_FLAGS )
FormatPassdowns( "MATLAB" VIAME_MATLAB_FLAGS )

ExternalProject_Add(viame
  DEPENDS ${VIAME_PROJECT_LIST}
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}
    ${VIAME_ARGS_scallop_tk}
    ${VIAME_ENABLE_FLAGS}
    ${VIAME_DISABLE_FLAGS}
    ${VIAME_MATLAB_FLAGS}
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DVIAME_BUILD_DEPENDENCIES:BOOL=OFF
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

ExternalProject_Add_Step(viame forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/viame-stamp/viame-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
