# external dependency for darknet

option( KWIVER_ENABLE_DARKNET
  "Enable darkent dependent code and plugins"
  ${fletch_ENABLED_Darknet}
  )
# Mark this as advanced until Darknet is provided by Fletch
mark_as_advanced( KWIVER_ENABLE_DARKNET )

if (KWIVER_ENABLE_DARKNET)
  find_package(Darknet)
  include_directories( SYSTEM ${Darknet_INCLUDE_DIR})
  include(kwiver-depends-Boost)
endif()
