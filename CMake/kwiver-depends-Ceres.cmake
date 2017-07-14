# Optionally find and configure Ceres dependency

option( KWIVER_ENABLE_CERES
  "Enable Ceres dependent code and plugins (Arrows)"
  ${fletch_ENABLED_Ceres}
  )

if( KWIVER_ENABLE_CERES )
  find_package( Ceres 1.10.0 REQUIRED )
  include_directories( SYSTEM ${CERES_INCLUDE_DIRS} )
endif()
