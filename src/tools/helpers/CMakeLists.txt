project(vistk_tools_helpers)

vistk_add_helper_library(tool_usage
  ${Boost_PROGRAM_OPTIONS_LIBRARY})

add_dependencies(tool_usage
  configure-version.h)

vistk_add_helper_library(pipeline_builder
  tool_usage
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
