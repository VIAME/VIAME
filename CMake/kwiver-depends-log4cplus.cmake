#
# Optionally find and configure log4cplus
#
option( KWIVER_ENABLE_LOG4CPLUS
  "Enable log4cplus dependent code and plugins"
  OFF
  )

if (KWIVER_ENABLE_LOG4CPLUS)

  find_package (Log4cplus REQUIRED)

else (KWIVER_ENABLE_LOG4CPLUS)

  unset ( Log4cplus_DIR         CACHE )
  unset ( Log4cplus_FOUND       CACHE )
  unset ( Log4cplus_INCLUDE_DIR CACHE )
  unset ( Log4cplus_LIBRARY     CACHE )

endif (KWIVER_ENABLE_LOG4CPLUS)
