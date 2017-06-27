# Optionally find and configure PROJ dependency

option( KWIVER_ENABLE_PROJ
  "Enable PROJ dependent code and plugins (Arrows)"
  ${fletch_ENABLED_PROJ4}
  )

# Fletch provides the PROJ4 symbols. We need the PROJ symbols.
if (PROJ4_ROOT AND NOT PROJ_ROOT)
  set(PROJ_ROOT "${PROJ4_ROOT}")
endif()

if( KWIVER_ENABLE_PROJ )
  find_package( PROJ REQUIRED )
  include_directories( SYSTEM ${PROJ_INCLUDE_DIR} )
endif( KWIVER_ENABLE_PROJ )
