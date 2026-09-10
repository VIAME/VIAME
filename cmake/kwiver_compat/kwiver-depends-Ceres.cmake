# Optionally find and configure Ceres dependency

kwiver_package_option(CERES
  DESCRIPTION "Enable Ceres dependent code and plugins (Arrows)"
  FLETCH_NAME Ceres
)

if( kwiver_enabled_ceres )
  find_package( Ceres REQUIRED )
  if(NOT CERES_INCLUDE_DIRS)
    # Ceres v2 doesn't define CERES_INCLUDE_DIRS, but the ceres algo plugin still relies on it
    get_target_property(CERES_INCLUDE_DIRS Ceres::ceres INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  include_directories( SYSTEM ${CERES_INCLUDE_DIRS} )
endif()
