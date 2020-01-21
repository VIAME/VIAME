# Build / Install plugin containing Qt support structures

set( CMAKE_AUTOMOC ON )

set( qt_headers_public
  image_container.h
  image_io.h
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
  image_io.cxx
  )

if( KWIVER_ENABLE_PROCESSES )
  list(APPEND qt_headers_public
    EmbeddedPipelineWorker.h
    )

  list(APPEND qt_sources
    EmbeddedPipelineWorker.cxx
    )
endif()

kwiver_add_library( kwiver_algo_qt
  ${qt_headers_public}
  ${qt_sources}
  )
target_link_libraries( kwiver_algo_qt
  PUBLIC  kwiver_algo_core
          Qt5::Gui
          Qt5::Core
  )
if( KWIVER_ENABLE_PROCESSES )
  target_link_libraries( kwiver_algo_qt
    PUBLIC  kwiver_adapter
            sprokit_pipeline
    )
endif()

algorithms_create_plugin( kwiver_algo_qt
  register_algorithms.cxx
  )

if( KWIVER_ENABLE_TESTS )
  add_subdirectory( tests )
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
