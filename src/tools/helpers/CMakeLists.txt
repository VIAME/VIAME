project(vistk_tools_helpers)

vistk_add_helper_library(pipeline_builder
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

vistk_add_helper_library(tool_usage
  ${Boost_PROGRAM_OPTIONS_LIBRARY})
