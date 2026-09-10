#
# Optionally find uuid library
#
option( KWIVER_ENABLE_UUID
  "Enable UUID dependent code and plugins"
  OFF
  )
# Mark as advanced until UUID is provided in Fletch
mark_as_advanced( KWIVER_ENABLE_UUID )

if (KWIVER_ENABLE_UUID)

  # Need some version of uuid library.
  # This is not optimal way of enabling/disabling UUID
  find_library( KWIVER_UUID_LIBRARY uuid )
  if( NOT KWIVER_UUID_LIBRARY )
    message( SEND_ERROR  "UUID library not found." )
  endif()

endif()
