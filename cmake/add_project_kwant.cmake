# kwant External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} kwant )

ExternalProject_Add(kwant
  DEPENDS fletch kwiver
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/kwant
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

VIAME_ExternalProject_Add_Step_Forcebuild(kwant)

set(VIAME_ARGS_kwant
  -Dkwant_DIR:PATH=${VIAME_BUILD_PREFIX}/src/kwant-build
  )
