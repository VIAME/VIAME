# The applet that prints a pipeline's configuration. Imported from
# kwiver's `vital/applets_plugins` in P5-T05; the sources are unchanged apart
# from naming their own generated export header and each other as siblings
# rather than through kwiver's binary root.
viame_add_plugin(
  config_explorer
  KIND        APPLETS
  SOURCES register_tools.cxx
          config_explorer.h
          config_explorer.cxx
  PRIVATE vital
)

target_include_directories( config_explorer PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
