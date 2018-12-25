# Build / Install plugin containing Qt support structures

set( qt_headers_public
  image_container.h
  )

kwiver_install_headers(
  SUBDIR     arrows/qt
  ${qt_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_qt_export.h
  NOPATH   SUBDIR     arrows/qt
  )

set( qt_sources
  image_container.cxx
  )

kwiver_add_library( kwiver_algo_qt
  ${qt_headers_public}
  ${qt_sources}
  )
target_link_libraries( kwiver_algo_qt
  PUBLIC  kwiver_algo_core
          Qt5::Gui
          Qt5::Core
  )
