# Optionally find and configure PROJ dependency

option( ARROWS_ENABLE_PROJ
  "Enable PROJ dependent code and plugins"
  OFF
  )

if( ARROWS_ENABLE_PROJ )
  set( old_prefix CMAKE_PREFIX_PATH )
  set( CMAKE_PREFIX_PATH ${PROJ4_ROOT} )

  find_package( PROJ4   MODULE REQUIRED )

  set( CMAKE_PREFIX_PATH   old_prefix )

  include_directories( SYSTEM ${PROJ_INCLUDE_DIR} )
endif( ARROWS_ENABLE_PROJ )
