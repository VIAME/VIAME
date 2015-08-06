# Optionally find and configure VisCL dependency

option( ${CMAKE_PROJECT_NAME}_ENABLE_VISCL
  "Enable VidCL dependent code and plugins"
  OFF
  )

if( ${CMAKE_PROJECT_NAME}_ENABLE_VISCL )
  find_package( viscl REQUIRED )
  include_directories( SYSTEM ${viscl_INCLUDE_DIR} )
  include_directories( SYSTEM ${viscl_OPENCL_INCLUDE_DIRS} )
endif( ${CMAKE_PROJECT_NAME}_ENABLE_VISCL )
