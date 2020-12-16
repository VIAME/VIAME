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
  set(Qt_components Core Gui)

  option( KWIVER_ENABLE_QT_EXT
    "Enable Qt Extensions dependent code"
    ${fletch_ENABLED_qtExtensions}
    )

  if( KWIVER_ENABLE_QT_EXT )
    list(APPEND Qt_components Widgets Xml)
    find_package(qtExtensions REQUIRED)
    include(${qtExtensions_USE_FILE})
  endif()

  find_package( Qt5 5.10 REQUIRED COMPONENTS ${Qt_components})

endif( KWIVER_ENABLE_QT )
