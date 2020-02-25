#
#
#
if ( KWIVER_ENABLE_PYTHON AND NOT WIN32 )
  add_subdirectory(python)
endif ()

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
