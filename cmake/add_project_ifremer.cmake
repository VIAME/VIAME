set(VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} ifremer_tk)

ExternalProject_Add(ifremer_tk
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/ifremer-tk
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
  ${VIAME_ARGS_COMMON}
  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

if( VIAME_FORCEBUILD )
  RemoveProjectCMakeStamp( ifremer_tk )
endif()

set(VIAME_ARGS_ifremer_tk
  -Difremer_tk_DIR:PATH=${VIAME_BUILD_PREFIX}/src/ifremer_tk-build
  )
