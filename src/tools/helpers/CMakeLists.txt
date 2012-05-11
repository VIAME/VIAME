project(vistk_tools_helpers)

add_library(pipeline_builder STATIC
  pipeline_builder.cxx)
target_link_libraries(pipeline_builder
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

if (NOT WIN32)
  set_target_properties(pipeline_builder
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)
