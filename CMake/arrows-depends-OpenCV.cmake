# Optionally find and configure OpenCV dependency

option( ARROWS_ENABLE_OPENCV
  "Enable OpenCV dependent code and plugins"
  OFF
  )

if( ARROWS_ENABLE_OPENCV )
  find_package( OpenCV REQUIRED )
  include_directories( SYSTEM ${OpenCV_INCLUDE_DIRS} )
  # Docs say we don't need to add link_directories() call for OpenCV lib dirs
endif()
