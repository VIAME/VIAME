# viame External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

function( formatPassdowns _prefix _varResult )
  get_cmake_property( _vars VARIABLES )
  string( REGEX MATCHALL "(^|;)${_prefix}[A-Za-z0-9_]*" _matchedVars "${_vars}" )
  set( ${_varResult} ${_matchedVars} PARENT_SCOPE )
endfunction()

formatPassdowns( "ENABLE" VIAME_ENABLE_FLAGS )
formatPassdowns( "DISABLE" VIAME_DISABLE_FLAGS )

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
    -DVIAME_BUILD_DEPENDENCIES:BOOL=OFF
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  INSTALL_COMMAND ""
  )

ExternalProject_Add_Step(viame forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/viame-stamp/viame-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
