project(vistk_tools)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    program_options)

add_executable(pipe_to_dot
  pipe_to_dot.cxx)
target_link_libraries(pipe_to_dot
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
