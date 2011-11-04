project(vistk_tools)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    program_options)

if (WIN32)
  add_definitions(-DBOOST_PROGRAM_OPTIONS_DYN_LINK)
endif (WIN32)

add_subdirectory(helpers)

add_executable(pipe_to_dot
  pipe_to_dot.cxx)
target_link_libraries(pipe_to_dot
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_executable(pipeline_runner
  pipeline_runner.cxx)
target_link_libraries(pipeline_runner
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_executable(processopedia
  processopedia.cxx)
target_link_libraries(processopedia
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
