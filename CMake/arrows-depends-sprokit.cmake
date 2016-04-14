#
# Optional find and configure sprokit dependency
#

option( ARROWS_ENABLE_SPROKIT
  "Enable sprokit dependent code and plugins"
  ON
  )

if ( ARROWS_ENABLE_SPROKIT )
  include(CommonFindMacros)

  find_package( sprokit REQUIRED )

  include_directories( SYSTEM ${SPROKIT_INCLUDE_DIRS} )
  link_directories( ${SPROKIT_LIBRARY_DIRS} )

  setup_find_root_context(BOOST)

  find_package( Boost ${SPROKIT_BOOST_VERSION} REQUIRED
    COMPONENTS
      chrono
      date_time
      filesystem
      program_options
      system
      thread)


  # Required for Boost.Thread.
  find_package(Threads REQUIRED)

  restore_find_root_context(EIGEN3)

  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})

endif( ARROWS_ENABLE_SPROKIT)
