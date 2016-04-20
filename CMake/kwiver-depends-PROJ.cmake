# Optionally find and configure PROJ dependency

option( KWIVER_ENABLE_PROJ
  "Enable PROJ dependent code and plugins"
  OFF
  )

if( KWIVER_ENABLE_PROJ )
  find_package( PROJ REQUIRED )
  include_directories( SYSTEM ${PROJ_INCLUDE_DIR} )
endif( KWIVER_ENABLE_PROJ )
