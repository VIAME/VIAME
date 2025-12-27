# Seal-TK External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} seal_tk )

if( MSVC )
  set( SEAL_CXX_FLAGS "/I${VIAME_BUILD_INSTALL_PREFIX}/include ${CMAKE_CXX_FLAGS}" )
else()
  set( SEAL_CXX_FLAGS -I${VIAME_BUILD_INSTALL_PREFIX}/include ${CMAKE_CXX_FLAGS} )
endif()

if( VIAME_ENABLE_VXL )
  set( SEAL_VIDEO_SOURCE vxl )
else()
  set( SEAL_VIDEO_SOURCE ocv )
endif()

ExternalProject_Add( seal_tk
  DEPENDS fletch kwiver
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/seal-tk
  USES_TERMINAL_BUILD 1
  CMAKE_GENERATOR ${gen}
  CMAKE_CACHE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_ARGS_fletch}
    ${VIAME_ARGS_kwiver}
    ${VIAME_ARGS_Qt}
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DNOAA_TRACK_READER:STRING=viame_csv
    -DNOAA_TRACK_WRITER:STRING=viame_csv
    -DNOAA_VIDEO_READER:STRING=add_timestamp_from_filename
    -DNOAA_VIDEO_READER_PASSTRHOUGH:STRING=${SEAL_VIDEO_SOURCE}
    -DCMAKE_CXX_FLAGS:STRING=${SEAL_CXX_FLAGS}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  )

if( VIAME_FORCEBUILD )
  RemoveProjectCMakeStamp( seal_tk )
endif()

set( VIAME_ARGS_seal_tk
  -Dseal_tk_DIR:PATH=${VIAME_BUILD_PREFIX}/src/seal_tk-build
  )

