#
# Optionally find and configure ZeroMQ dependency

option( KWIVER_ENABLE_ZeroMQ
  "Enable ZeroMQ dependent code and plugins"
  OFF
  )

if( KWIVER_ENABLE_ZeroMQ )
  if(KWIVER_BUILD_SHARED)
    find_package( ZeroMQ REQUIRED )
  else()
    find_library( ZeroMQ_LIBRARY libzmq.a libzmq
                  PATHS ${fletch_ROOT}
                  PATH_SUFFIXES lib
                )
  endif()
  include_directories(SYSTEM ${ZeroMQ_INCLUDE_DIR})
endif( KWIVER_ENABLE_ZeroMQ )
