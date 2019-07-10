# Optionally find and configure Qt dependency

if( Qt5_DIR )
  set( KWIVER_ENABLE_QT_DEFAULT ON )
else()
  set( KWIVER_ENABLE_QT_DEFAULT OFF )
endif()

option( KWIVER_ENABLE_QT
  "Enable Qt dependent code and plugins (Arrows)"
  ${KWIVER_ENABLE_QT_DEFAULT}
  )

if( KWIVER_ENABLE_QT )
  find_package( Qt5 5.10 REQUIRED COMPONENTS Core Gui )
endif( KWIVER_ENABLE_QT )
