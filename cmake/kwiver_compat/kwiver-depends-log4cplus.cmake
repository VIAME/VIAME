#
# Optionally find and configure log4cplus
#

kwiver_package_option(LOG4CPLUS
  DESCRIPTION "Enable log4cplus dependent code for a Vital logger plugin"
  FLETCH_NAME Log4cplus
)

if( kwiver_enabled_log4cplus )

  find_package (log4cplus REQUIRED)

else()

  unset ( log4cplus_DIR         CACHE )
  unset ( log4cplus_FOUND       CACHE )
  unset ( log4cplus_INCLUDE_DIR CACHE )
  unset ( log4cplus_LIBRARY     CACHE )

endif()
