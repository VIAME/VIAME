# Optional find and configure depth dependency

option( KWIVER_ENABLE_DEPTH
  "Enable DEPTH dependent code and plugins (Arrows)"
  ${fletch_ENABLED_DEPTH}
  )

if( KWIVER_ENABLE_DEPTH )
  find_package( VXL REQUIRED )
  include(${VXL_CMAKE_DIR}/UseVXL.cmake)
  include_directories( SYSTEM ${VXL_CORE_INCLUDE_DIR} )
  link_directories( ${VXL_LIBRARY_DIR} )
endif( KWIVER_ENABLE_DEPTH )
