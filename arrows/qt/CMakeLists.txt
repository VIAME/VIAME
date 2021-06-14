# Build / Install plugin containing Qt support structures

set(CMAKE_FOLDER "Arrows/Qt")

set( CMAKE_AUTOMOC ON )
set( CMAKE_AUTOUIC ON )
set( CMAKE_INCLUDE_CURRENT_DIR ON )

set( qt_sources
  image_container.cxx
  image_io.cxx
  )

set( qt_headers_public
  image_container.h
  image_io.h
  kq_global.h
  )

set( qt_libraries_public
  kwiver_algo_core
  Qt5::Gui
  Qt5::Core
  )

if( KWIVER_ENABLE_PROCESSES )
  list(APPEND qt_sources
    EmbeddedPipelineWorker.cxx
  )
  list(APPEND qt_headers_public
    EmbeddedPipelineWorker.h
  )
  list(APPEND qt_libraries_public
    kwiver_adapter
    sprokit_pipeline
  )
endif()

kwiver_add_library( kwiver_algo_qt
  ${qt_headers_public}
  ${qt_sources}
  )
target_link_libraries( kwiver_algo_qt
  PUBLIC ${qt_libraries_public}
  )

algorithms_create_plugin( kwiver_algo_qt
  register_algorithms.cxx
  )

kwiver_install_headers(
  SUBDIR     arrows/qt
  ${qt_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_qt_export.h
  NOPATH   SUBDIR     arrows/qt
  )

if( KWIVER_ENABLE_QT_EXT )
  add_subdirectory( widgets )
endif()

if( KWIVER_ENABLE_TESTS )
  add_subdirectory( tests )
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
