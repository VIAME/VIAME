find_package( Qt5 REQUIRED COMPONENTS Core Gui Widgets )

set( CMAKE_AUTOMOC ON )
set( CMAKE_AUTOUIC ON )
set( CMAKE_INCLUDE_CURRENT_DIR ON )

set( sources
  ImageView.cxx
  MainWindow.cxx
  PipelineWorker.cxx
  pipeline_viewer.cxx
  register_tools.cxx
  )

set( headers
  ImageView.h
  MainWindow.h
  PipelineWorker.h
  pipeline_viewer.h
  )

kwiver_add_plugin( kwiver_pipeline_viewer
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${sources} ${headers}
  PRIVATE      kwiver_tools_applet
               kwiver_algo_qt
               Qt5::Widgets
  )
