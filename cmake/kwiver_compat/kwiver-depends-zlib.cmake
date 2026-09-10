# Optional find and configure ZLib dependency

kwiver_package_option(ZLIB
  DESCRIPTION "Enable zlib dependent code and plugins (Arrows)"
  FLETCH_NAME ZLib
)

if( kwiver_enabled_zlib )
  find_package( ZLIB MODULE REQUIRED )
endif()
