
set( VIAME_DIVE_BUILD_DIR ${VIAME_BUILD_PREFIX}/build/src/dive-build )
file( MAKE_DIRECTORY ${VIAME_DIVE_BUILD_DIR} )

if( WIN32 )
  DownloadExtractAndInstall(
    https://github.com/VIAME/VIAME-Web/releases/download/1.4.1/DIVE-Desktop-1.4.1.zip
    6ea9b6c226575ef9dc6cbd0085e37908
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.zip
    ${VIAME_DIVE_BUILD_DIR}
    bin/dive )
elseif( UNIX )
  DownloadExtractAndInstall(
    https://github.com/VIAME/VIAME-Web/releases/download/1.4.1/DIVE-Desktop-1.4.1.tar.gz
    5ed66a76cde3acb556f8b5a603da66cd
    ${VIAME_DOWNLOAD_DIR}/dive_interface_binaries.tar.gz
    ${VIAME_DIVE_BUILD_DIR}
    bin/dive )
endif()
