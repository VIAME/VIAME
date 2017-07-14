# Optionally find and configure VisCL dependency

option( KWIVER_ENABLE_VISCL
  "Enable VidCL dependent code and plugins"
  OFF
  )
mark_as_advanced( KWIVER_ENABLE_VISCL )

if( KWIVER_ENABLE_VISCL )
  find_package( viscl REQUIRED )
  include_directories( SYSTEM ${viscl_INCLUDE_DIR} )
  include_directories( SYSTEM ${viscl_OPENCL_INCLUDE_DIRS} )
endif( KWIVER_ENABLE_VISCL )
