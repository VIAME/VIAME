# VIAME Internal Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

FormatPassdowns( "VIAME" VIAME_VIAME_FLAGS )

if( VIAME_ENABLE_MATLAB )
  FormatPassdowns( "Matlab" VIAME_MATLAB_FLAGS )
endif()

if( VIAME_ENABLE_PYTHON )
  FormatPassdownsCS( "Python" VIAME_PYTHON_FLAGS )
endif()

ExternalProject_Add(viame
  DEPENDS ${VIAME_PROJECT_LIST}
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}
  BINARY_DIR ${VIAME_PLUGINS_BUILD_DIR}
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}
    ${VIAME_ARGS_scallop_tk}
    ${VIAME_ARGS_ITK}
    ${VIAME_VIAME_FLAGS}
    ${VIAME_MATLAB_FLAGS}
    ${VIAME_PYTHON_FLAGS}
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DVIAME_BUILD_DEPENDENCIES:BOOL=OFF
    -DVIAME_IN_SUPERBUILD:BOOL=ON
    -DKWIVER_PYTHON_MAJOR_VERSION:STRING=${Python_VERSION_MAJOR}
  INSTALL_DIR ${VIAME_INSTALL_PREFIX}
  )

#if ( VIAME_FORCEBUILD )
ExternalProject_Add_Step(viame forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/viame-stamp/viame-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )
#endif()
