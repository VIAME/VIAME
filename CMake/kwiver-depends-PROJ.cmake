# Optionally find and configure PROJ dependency

option( ${CMAKE_PROJECT_NAME}_ENABLE_PROJ
  "Enable PROJ dependent code and plugins"
  OFF
  )

if( ${CMAKE_PROJECT_NAME}_ENABLE_PROJ )
  find_package( PROJ REQUIRED )
  include_directories( SYSTEM ${PROJ_INCLUDE_DIR} )
endif( ${CMAKE_PROJECT_NAME}_ENABLE_PROJ )
