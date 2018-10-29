#
# Optionally find and configure ZeroMQ dependency

option( KWIVER_ENABLE_ZeroMQ
  "Enable ZeroMQ dependent code and plugins"
  OFF
  )

if( KWIVER_ENABLE_ZeroMQ )
  find_package( ZeroMQ REQUIRED )
  include_directories(SYSTEM ${ZeroMQ_INCLUDE_DIR})
endif( KWIVER_ENABLE_ZeroMQ )
