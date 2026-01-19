# ITK External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} itk_module_tps )

ExternalProject_Add( itk_module_tps
  DEPENDS fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/itk-modules/trimmed-point-set
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_VXL}
    ${VIAME_ARGS_ITK}
    -DBUILD_TESTING:BOOL=OFF
  INSTALL_DIR ${VIAME_INSTALL_PREFIX}
)

if( VIAME_BUILD_FORCE_REBUILD )
  RemoveProjectCMakeStamp( itk_module_tps )
endif()
