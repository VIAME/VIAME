# Optional Boost external dependency
if (NOT DEFINED KWIVER_BOOST_VERSION)
  set(KWIVER_BOOST_VERSION 1.54)
endif()

find_package(Boost ${KWIVER_BOOST_VERSION}
  COMPONENTS
  chrono
  date_time
  iostreams
  filesystem
  program_options
  regex
  system
  thread)

if(Boost_FOUND)
  add_definitions(-DBOOST_ALL_NO_LIB)

  if(WIN32)
    add_definitions(-DBOOST_PROGRAM_OPTIONS_DYN_LINK)
  endif()

  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif()
####
