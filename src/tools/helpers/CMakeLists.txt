project(vistk_tools_helpers)

function (add_tool_helper_library name)
  vistk_add_helper_library(${name}
    ${ARGN})
  if (boost_header_targets)
    add_dependencies(${name}
      ${boost_header_targets})
  endif ()
endfunction ()

add_tool_helper_library(tool_usage
  ${Boost_PROGRAM_OPTIONS_LIBRARY})

add_dependencies(tool_usage
  configure-version.h)

add_tool_helper_library(tool_io
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool_helper_library(pipeline_builder
  tool_usage
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
