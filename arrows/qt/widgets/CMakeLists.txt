# Build / Install plugin containing Qt widgets

set( qt_sources
  LoggerView.cxx
  MetadataView.cxx
  )

set( qt_headers_public
  LoggerView.h
  MetadataView.h
  )

set( qt_libraries_public
  Qt5::Widgets
  Qt5::Xml
  qtExtensions
  vital
  )

kwiver_add_library( kwiver_algo_qt_widgets
  ${qt_headers_public}
  ${qt_sources}
  )
target_link_libraries( kwiver_algo_qt_widgets
  PUBLIC ${qt_libraries_public}
  )

kwiver_install_headers(
  SUBDIR     arrows/qt/widgets
  ${qt_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_qt_widgets_export.h
  NOPATH   SUBDIR     arrows/qt/widgets
  )
