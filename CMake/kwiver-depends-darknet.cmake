# external dependency for darknet

option( KWIVER_ENABLE_DARKNET
  "Enable darkent dependent code and plugins"
  OFF
  )
# Mark this as advanced until Darknet is provided by Fletch
mark_as_advanced( KWIVER_ENABLE_DARKNET )

if (KWIVER_ENABLE_DARKNET)
  find_package(darknet)
  include_directories( SYSTEM ${darknet_INCLUDE_DIR})
  link_directories( ${darknet_LIBRARY_DIR} )
endif()
