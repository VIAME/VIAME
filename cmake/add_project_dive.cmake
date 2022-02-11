
set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadAndExtract(
    https://github.com/Kitware/dive/releases/download/1.7.7/DIVE-Desktop-1.7.7.zip
    fb51d288848a73b3afdb5f4924ef9c4c
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR} )
elseif( UNIX )
  DownloadAndExtract(
    https://github.com/Kitware/dive/releases/download/1.7.7/DIVE-Desktop-1.7.7.tar.gz
    0579e0e32df9cc4f4bd662913027a36b
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
