if( WIN32 AND MSVC )
  CopyFiles( ${CURRENT_BINARY_DIR}/../kwiver-build/bin/*/kwiversys.dll ${VIAME_BUILD_INSTALL_PREFIX}/bin )
endif()