
set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/bin/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadAndExtract(
    https://github.com/VIAME/VIAME-Web/releases/download/1.4.1/DIVE-Desktop-1.4.1.zip
    6ea9b6c226575ef9dc6cbd0085e37908
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR} )
elseif( UNIX )
  DownloadAndExtract(
    https://github.com/VIAME/VIAME-Web/releases/download/1.4.1/DIVE-Desktop-1.4.1.tar.gz
    5ed66a76cde3acb556f8b5a603da66cd
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
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
