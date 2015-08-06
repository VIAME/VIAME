# Optionally find and configure OpenCV dependency

option( ${CMAKE_PROJECT_NAME}_ENABLE_OPENCV
  "Enable OpenCV dependent code and plugins"
  OFF
  )

set( USE_OPENCV False )
if( ${CMAKE_PROJECT_NAME}_ENABLE_OPENCV )
  find_package( OpenCV 2.4.6 REQUIRED )
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

  if( OpenCV_VERSION VERSION_GREATER "2.4" )
    set( USE_OPENCV True )
  else()
    message( FATAL_ERROR "OpenCV version must be at least 2.4" )
  endif()
endif( ${CMAKE_PROJECT_NAME}_ENABLE_OPENCV )

  ##
