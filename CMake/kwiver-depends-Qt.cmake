# Optionally find and configure Qt dependency

option( KWIVER_ENABLE_QT
  "Enable Qt dependent code and plugins (Arrows)"
  ${fletch_ENABLED_Qt}
  )

if( KWIVER_ENABLE_QT )
  find_package( Qt5 REQUIRED COMPONENTS Core Gui )
endif( KWIVER_ENABLE_QT )
