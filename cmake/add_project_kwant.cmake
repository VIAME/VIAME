# kwant External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

if( VIAME_ENABLE_PYTHON )
  FormatPassdowns( "Python" VIAME_PYTHON_FLAGS )
endif()

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} kwant )

ExternalProject_Add(kwant
  DEPENDS fletch kwiver
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/kwant
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}
    ${VIAME_PYTHON_FLAGS}

  INSTALL_DIR ${VIAME_INSTALL_PREFIX}
  )

if( VIAME_BUILD_FORCE_REBUILD )
  RemoveProjectCMakeStamp( kwant )
endif()

set(VIAME_ARGS_kwant
  -Dkwant_DIR:PATH=${VIAME_BUILD_PREFIX}/src/kwant-build
  )
