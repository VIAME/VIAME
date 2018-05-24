# Optionally find and configure GDAL dependency

option( KWIVER_ENABLE_GDAL
  "Enable GDAL dependent code and plugins (Arrows)"
  ${fletch_ENABLE_GDAL}
  )

if( KWIVER_ENABLE_GDAL )
  find_package( GDAL REQUIRED )
  if( GDAL_FOUND )
    if( GDAL_CONFIG )
      EXEC_PROGRAM(${GDAL_CONFIG}
        ARGS --version
        OUTPUT_VARIABLE GDAL_VERSION )
      STRING(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\1" GDAL_VERSION_MAJOR "${GDAL_VERSION}")
      STRING(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\2" GDAL_VERSION_MINOR "${GDAL_VERSION}")

      message(STATUS "Found GDAL ${GDAL_VERSION}")

      if( GDAL_VERSION_MAJOR LESS 2 OR GDAL_VERSION_MINOR LESS 3 )
        message( FATAL_ERROR "GDAL ${GDAL_VERSION} found, but version must be 2.3.0 or higher." )
        unset(GDAL_INCLUDE_DIR)
      else()
        include_directories(SYSTEM ${GDAL_INCLUDE_DIR})
      endif()
    else()
      message( FATAL_ERROR "gdal-config missing from GDAL installation. Can not test for GDAL version." )
      unset(GDAL_INCLUDE_DIR)
    endif()
  endif()
endif()
