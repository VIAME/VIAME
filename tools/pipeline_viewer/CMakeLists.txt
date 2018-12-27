find_package( Qt5 REQUIRED COMPONENTS Core Gui Widgets )

set( CMAKE_AUTOMOC ON )
set( CMAKE_AUTOUIC ON )
set( CMAKE_INCLUDE_CURRENT_DIR ON )

set( sources
  ImageView.cxx
  MainWindow.cxx
  main.cxx
  PipelineWorker.cxx
  )

set( headers
  ImageView.h
  MainWindow.h
  PipelineWorker.h
  )

add_executable( pipeline_viewer
  ${sources}
  ${headers}
  )

target_link_libraries( pipeline_viewer
  PRIVATE
  kwiver_algo_qt
  Qt5::Widgets
  )
