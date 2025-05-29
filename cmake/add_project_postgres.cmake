# Postgres is provided via fletch for only linux but not windows
if( WIN32 )
  include( ExternalProject )

  set( VIAME_POSTGRES_URL https://viame.kitware.com/api/v1/file/6837c74a1c2c16143e306f40/download )
  set( VIAME_POSTGRES_MD5 5e01e77deb12cd7f0c4df24903694098 )
  set( VIAME_POSTGRES_FILENAME postgresql-10.23-1-windows-x64-binaries.zip )

  set( VIAME_POSTGRES_DIR ${VIAME_BUILD_PREFIX}/src/postgres )

  ExternalProject_Add( postgres
    DEPENDS fletch
    URL ${VIAME_POSTGRES_URL}
    URL_MD5 ${VIAME_POSTGRES_MD5}
    DOWNLOAD_DIR ${VIAME_DOWNLOAD_DIR}
    DOWNLOAD_NAME ${VIAME_POSTGRES_FILENAME}
    PREFIX ${VIAME_BUILD_PREFIX}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${VIAME_POSTGRES_DIR}/bin ${VIAME_BUILD_INSTALL_PREFIX}/bin
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${VIAME_POSTGRES_DIR}/lib ${VIAME_BUILD_INSTALL_PREFIX}/lib
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${VIAME_POSTGRES_DIR}/share ${VIAME_BUILD_INSTALL_PREFIX}/share
    )
endif()
