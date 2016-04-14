# Optionally find and configure VisCL dependency

option( ARROWS_ENABLE_VISCL
  "Enable VisCL dependent code and plugins"
  OFF
  )

if( ARROWS_ENABLE_VISCL )
  find_package( viscl REQUIRED )
  include_directories( SYSTEM ${viscl_INCLUDE_DIR} )
  include_directories( SYSTEM ${viscl_OPENCL_INCLUDE_DIRS} )
endif( ARROWS_ENABLE_VISCL )
