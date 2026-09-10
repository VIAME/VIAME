# Provide simple abstract interface for plugin provision testing

# plugin module with simple C++-side implementation
kwiver_add_plugin(
  say_cpp
  SOURCES plugin.cxx
  PRIVATE vital
  SUBDIR  ${kwiver_plugin_module_subdir}
)

target_include_directories( say_cpp PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
