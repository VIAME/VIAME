# Optional confgure DBoW2 dependency

kwiver_package_option(DBOW2
  DESCRIPTION "Enable DBoW2 dependent code and plugins"
  DEFAULT ${kwiver_enabled_opencv}
)

if( kwiver_enabled_dbow2 )
  find_package( OpenCV REQUIRED )
endif()
