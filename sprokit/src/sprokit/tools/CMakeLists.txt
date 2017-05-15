project(sprokit_tools)

sprokit_use_tools_fixes()

set(tools_srcs
  tool_io.cxx
  tool_usage.cxx
  build_pipeline_from_options.cxx
  )

set(tools_headers
  tool_io.h
  tool_main.h
  tool_usage.h
  build_pipeline_from_options.h
  )

kwiver_add_library(sprokit_tools
  ${tools_srcs}
  ${tools_headers})

target_link_libraries(sprokit_tools
  PUBLIC       sprokit_pipeline_util
               sprokit_pipeline
               ${Boost_PROGRAM_OPTIONS_LIBRARY}

  PRIVATE      ${Boost_FILESYSTEM_LIBRARY}
               ${Boost_SYSTEM_LIBRARY}
               vital_config
  )

sprokit_require_tools_fixes(sprokit_tools)

kwiver_install_headers(
  SUBDIR sprokit/tools
  ${tools_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/sprokit_tools_export.h
  NOPATH   SUBDIR  sprokit/tools
  )

sprokit_configure_pkgconfig(sprokit-tools)
