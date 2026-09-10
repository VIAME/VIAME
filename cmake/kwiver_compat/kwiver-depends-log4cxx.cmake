#
# Optionally find and configure log4cxx
#
option( KWIVER_ENABLE_LOG4CXX
  "Enable log4cxx dependent code for a Vital logger plugin"
  OFF
  )
# Not supported by all platforms, so this is an advanced option
mark_as_advanced( KWIVER_ENABLE_LOG4CXX )

if (KWIVER_ENABLE_LOG4CXX)

  find_package (Log4cxx REQUIRED)
  # find_package (ApacheRunTime REQUIRED)

else (KWIVER_ENABLE_LOG4CXX)

  unset ( Log4cxx_DIR         CACHE )
  unset ( Log4cxx_FOUND       CACHE )
  unset ( Log4cxx_INCLUDE_DIR CACHE )
  unset ( Log4cxx_LIBRARY     CACHE )

  # unset ( ApacheRunTime_FOUND       CACHE )
  # unset ( ApacheRunTime_INCLUDE_DIR CACHE )
  # unset ( ApacheRunTime_LIBRARY     CACHE )

endif (KWIVER_ENABLE_LOG4CXX)
