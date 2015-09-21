# Required Boost external dependency

if(WIN32)
  set(Boost_USE_STATIC_LIBS TRUE)
  set(Boost_WIN_MODULES chrono)
endif()

find_package(Boost 1.55 REQUIRED
  COMPONENTS
    chrono
    date_time
    filesystem
    program_options
    regex
    system
    thread
    )

add_definitions(-DBOOST_ALL_NO_LIB)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
####
