
set( VIAME_DIVE_BUILD_DIR "${VIAME_BUILD_PREFIX}/src/dive-build" )
set( VIAME_DIVE_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}/dive" )

file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/68d1fa44af4f192121e81678/download
    2a1fda47f158cf083d19893b8b27598b
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR} )
elseif( UNIX )
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/68d1fa36af4f192121e81675/download
    41dcd42e9b1e7fca30917505ebaa6148
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
