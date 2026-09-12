# Provide simple abstract interface for plugin provision testing

# plugin module with simple C++-side implementation
viame_add_plugin(
  say_cpp
  SOURCES plugin.cxx
  PRIVATE vital
  KIND        OTHERS
)

target_include_directories( say_cpp PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
