message("Running fletch install")

include( ${VIAME_CMAKE_DIR}/common_macros.cmake )

if( WIN32 )

  if( MSVC AND MSVC_VERSION EQUAL 1900 )
    RenameSubstr( ${VIAME_BUILD_INSTALL_PREFIX}/lib/libboost* vc120 vc140 )
  endif()

  if( VIAME_ENABLE_OPENCV )
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x64/*/bin/*.dll ${VIAME_BUILD_INSTALL_PREFIX}/bin )
      CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x64/*/lib/*.lib ${VIAME_BUILD_INSTALL_PREFIX}/lib )
    else()
      CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x86/*/bin/*.dll ${VIAME_BUILD_INSTALL_PREFIX}/bin )
      CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x86/*/lib/*.lib ${VIAME_BUILD_INSTALL_PREFIX}/lib )
    endif()
  endif()

endif()

message("Done")
