# The input and output adapters
#
# What an embedded pipeline is driven through. Imported from kwiver's
# `sprokit/processes/adapters` in P5-T03 and built from here since P5-T05.

set( lib_sources
  adapter_data_set.cxx
  input_adapter.cxx
  output_adapter.cxx
  embedded_pipeline.cxx
  embedded_pipeline_extension.cxx
  adapter_base.h           adapter_base.cxx
  input_adapter_process.h  input_adapter_process.cxx
  output_adapter_process.h output_adapter_process.cxx
  )

set( public_headers
  adapter_types.h
  adapter_data_set.h
  input_adapter.h
  output_adapter.h
  embedded_pipeline.h
  embedded_pipeline_extension.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

viame_add_library( viame_adapter
  ${lib_sources}
  ${public_headers}
  )

target_link_libraries( viame_adapter
  PUBLIC           viame_algorithm_framework
                   viame_plugin
                   viame_logger
  PRIVATE          viame_pipeline_framework
                   viame_pipeline_util
  )

add_library( kwiver::kwiver_adapter ALIAS viame_adapter )

viame_add_plugin( viame_processes_adapter
  KIND        PROCESSES
  SOURCES         register_processes.cxx
  PRIVATE         viame_adapter viame_pipeline_framework
                  viame_config
)

viame_add_plugin( viame_epx_test
  KIND        PROCESSES
  SOURCES         epx_test.h
                  epx_test.cxx
                  ${CMAKE_CURRENT_BINARY_DIR}/viame_epx_test_export.h
  PUBLIC          viame_pipeline_framework
  PRIVATE         viame_adapter
                  viame_config
)

viame_lite_generated( viame_adapter_export.h pipeline_framework/adapters )
viame_lite_generated( viame_epx_test_export.h pipeline_framework/adapters )
viame_lite_generated( viame_processes_adapter_export.h pipeline_framework/adapters )
