# VIAME Plugin Tests

add_subdirectory( core )

if( VIAME_ENABLE_OPENCV )
  add_subdirectory( opencv )
endif()

if( VIAME_ENABLE_PYTORCH )
  add_subdirectory( pytorch )
endif()
