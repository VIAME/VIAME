# Required Boost external dependency

if (KWIVER_ENABLE_SPROKIT OR KWIVER_ENABLE_TRACK_ORACLE)

  if(WIN32)
    set(Boost_WIN_MODULES chrono)
  endif()

  find_package(Boost 1.55 REQUIRED
    COMPONENTS
      chrono
      date_time
      ${kwiver_boost_python_package}
      filesystem
      program_options
      regex
      system
      thread)

  add_definitions(-DBOOST_ALL_NO_LIB)

  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})

endif()
####
