# Optional find and configure depth dependency

option( KWIVER_ENABLE_SUPER3D
  "Enable SUPER3D dependent code and plugins (Arrows)"
  ON
  )

if( KWIVER_ENABLE_SUPER3D )
  find_package( VXL REQUIRED )
  include(${VXL_CMAKE_DIR}/UseVXL.cmake)
  include_directories( SYSTEM ${VXL_CORE_INCLUDE_DIR} )
  link_directories( ${VXL_LIBRARY_DIR} )
endif( KWIVER_ENABLE_SUPER3D )
