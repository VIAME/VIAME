# Optionally find and configure GDAL dependency

option( KWIVER_ENABLE_GDAL
  "Enable GDAL dependent code and plugins (Arrows)"
  ${fletch_ENABLE_GDAL}
  )

if( KWIVER_ENABLE_GDAL )
  find_package( GDAL REQUIRED )
  if( GDAL_FOUND )
    # We need to build the file in a line-by-line fashon because of
    # portability with end of line markers.
    file( WRITE  ${CMAKE_BINARY_DIR}/test_gdal_version.cxx "#include <gdal_version.h>\n" )
    file( APPEND ${CMAKE_BINARY_DIR}/test_gdal_version.cxx "#if ( GDAL_COMPUTE_VERSION( 2, 3, 0  ) != GDAL_VERSION_NUM )\n")
    file( APPEND ${CMAKE_BINARY_DIR}/test_gdal_version.cxx "#error \"GDAL Not required version: $GDAL_VERSION_NUM\"\n" )
    file( APPEND ${CMAKE_BINARY_DIR}/test_gdal_version.cxx "#endif\n")
    file( APPEND ${CMAKE_BINARY_DIR}/test_gdal_version.cxx "int main() { } // just need some code\n")

    TRY_COMPILE( GDAL_VERSION_MATCH
               ${CMAKE_BINARY_DIR}
               ${CMAKE_BINARY_DIR}/test_gdal_version.cxx
               COMPILE_DEFINITIONS "-I${GDAL_INCLUDE_DIR}"
               OUTPUT_VARIABLE OUTPUT)

    file( REMOVE ${CMAKE_BINARY_DIR}/test_gdal_version.cxx )

    if( GDAL_VERSION_MATCH )
      include_directories(SYSTEM ${GDAL_INCLUDE_DIR})
    else()
      message( FATAL_ERROR "GDAL found, but not needed version. Need version: 2.3.0" )
      message(${OUTPUT})
      unset(GDAL_INCLUDE_DIR)
    endif()
  endif()
endif()
