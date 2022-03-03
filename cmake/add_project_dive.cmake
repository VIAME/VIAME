
set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadAndExtract(
    https://github.com/Kitware/dive/releases/download/1.7.8/DIVE-Desktop-1.7.8.zip
    5ebb9087cced81f8c54967d1a98159bd
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR} )
elseif( UNIX )
  DownloadAndExtract(
    https://github.com/Kitware/dive/releases/download/1.7.8/DIVE-Desktop-1.7.8.tar.gz
    ef93d9ceb2204d34a3af4c727b310ced
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.tar.gz
    ${VIAME_DIVE_BUILD_DIR} )
endif()

if( WIN32 )
  file( GLOB ALL_DIR_FILES "${VIAME_DIVE_BUILD_DIR}/*" )
  file( COPY ${ALL_DIR_FILES} DESTINATION ${VIAME_DIVE_INSTALL_DIR} )
else()
  file( GLOB DIVE_SUBDIRS RELATIVE ${VIAME_DIVE_BUILD_DIR} ${VIAME_DIVE_BUILD_DIR}/DIVE* )
  foreach( SUBDIR ${DIVE_SUBDIRS} )
    if( IS_DIRECTORY ${VIAME_DIVE_BUILD_DIR}/${SUBDIR} )
      file( GLOB ALL_DIR_FILES "${VIAME_DIVE_BUILD_DIR}/${SUBDIR}/*" )
      file( COPY ${ALL_DIR_FILES} DESTINATION ${VIAME_DIVE_INSTALL_DIR} )
    endif()
  endforeach()
endif()
