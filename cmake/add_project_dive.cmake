
set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadAndExtract(
    https://github.com/Kitware/dive/releases/download/v1.9.7/DIVE-Desktop-1.9.7.zip
    b69ff0bb6e56a027d04a3e3b0fb6497e
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR} )
elseif( UNIX )
  DownloadAndExtract(
    https://github.com/Kitware/dive/releases/download/v1.9.7/DIVE-Desktop-1.9.7.tar.gz
    60f4894a3f01303832c4f71973cab6f2
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.tar.gz
    ${VIAME_DIVE_BUILD_DIR} )
endif()

file( GLOB DIVE_SUBDIRS RELATIVE ${VIAME_DIVE_BUILD_DIR} ${VIAME_DIVE_BUILD_DIR}/DIVE* )

if( DIVE_SUBDIRS )
  foreach( SUBDIR ${DIVE_SUBDIRS} )
    if( IS_DIRECTORY ${VIAME_DIVE_BUILD_DIR}/${SUBDIR} )
      file( GLOB ALL_FILES_IN_DIR "${VIAME_DIVE_BUILD_DIR}/${SUBDIR}/*" )
      file( COPY ${ALL_FILES_IN_DIR} DESTINATION ${VIAME_DIVE_INSTALL_DIR} )
    endif()
  endforeach()
else()
  file( GLOB ALL_FILES_IN_DIR "${VIAME_DIVE_BUILD_DIR}/*" )
  file( COPY ${ALL_FILES_IN_DIR} DESTINATION ${VIAME_DIVE_INSTALL_DIR} )
endif()
