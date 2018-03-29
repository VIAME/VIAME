# Optionally find and configure GDAL dependency

option( KWIVER_ENABLE_GDAL
  "Enable GDAL dependent code and plugins (Arrows)"
  ${fletch_ENABLE_GDAL}
  )

if( KWIVER_ENABLE_GDAL )
  find_package( GDAL REQUIRED )
  include_directories( SYSTEM ${GDAL_INCLUDE_DIRS} )
endif()
