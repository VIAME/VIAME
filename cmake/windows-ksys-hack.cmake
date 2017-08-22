if( WIN32 AND MSVC )
  CopyFiles( ${CMAKE_BINARY_DIR}/../kwiver-build/bin/*/kwiversys.dll ${CMAKE_BINARY_DIR}/bin )

  if( EXISTS ${CMAKE_BINARY_DIR}/bin/kwiversys.dll )
    install( FILES ${CMAKE_BINARY_DIR}/bin/kwiversys.dll DESTINATION bin )
  endif()
endif()