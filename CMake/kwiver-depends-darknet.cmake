# external dependency for darknet

option( KWIVER_ENABLE_DARKNET
  "Enable darkent dependent code and plugins"
  OFF
  )

if (KWIVER_ENABLE_DARKNET)
  find_package(darknet)
  include_directories( SYSTEM ${darknet_INCLUDE_DIR})
  if (WIN32)
    include_directories( SYSTEM ${darknet_INCLUDE_DIR}/../3rdparty/include)
  endif()
  link_directories( ${darknet_LIBRARY_DIR} )
endif()
