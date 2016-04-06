# Optionally find and configure PROJ dependency

option( ARROWS_ENABLE_PROJ
  "Enable PROJ dependent code and plugins"
  OFF
  )

if( ARROWS_ENABLE_PROJ )
  include(CommonFindMacros)
  setup_find_root_context(PROJ4)

  find_package( PROJ4   MODULE REQUIRED )

  restore_find_root_context(EIGEN3)

  include_directories( SYSTEM ${PROJ_INCLUDE_DIR} )
endif( ARROWS_ENABLE_PROJ )
