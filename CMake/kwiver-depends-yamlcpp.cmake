if(fletch_ENABLED_YAMLCPP)
  find_package( yaml-cpp REQUIRED )
  include_directories( SYSTEM ${YAMLCPP_INCLUDE_DIR} )
endif()
