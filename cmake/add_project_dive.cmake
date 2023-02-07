
set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/63d44a016d3fc641a02d8efa/download
    cae560a5e04c8a03204e7e0488390f19
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR} )
elseif( UNIX )
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/63d44bad6d3fc641a02d8efd/download
    68caa3752d3ebf6000c47504849d2ed2
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
