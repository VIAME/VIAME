# Find dependencies.

########################################
# Boost
########################################

set(sprokit_boost_version 1.47)

if (SPROKIT_ENABLE_PYTHON)
  set(sprokit_boost_version 1.48)
endif ()

# Required for Boost.Thread.
find_package(Threads REQUIRED)

set(BOOST_ROOT "" CACHE PATH "The root path to Boost")
option(Boost_USE_STATIC_LIBS "Use a statically-linked Boost" OFF)
find_package(Boost ${sprokit_boost_version} REQUIRED
  COMPONENTS
    chrono
    date_time
    filesystem
    system
    thread)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if (WIN32)
  add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

include("${sprokit_source_dir}/cmake/snippets/boost_tests.cmake")
