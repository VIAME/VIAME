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

viame_add_library( kwiver_adapter
  ${lib_sources}
  ${public_headers}
  )

target_link_libraries( kwiver_adapter
  PUBLIC           vital
                   vital_vpm
                   vital_logger
  PRIVATE          sprokit_pipeline
                   sprokit_pipeline_util
  )

add_library( kwiver::kwiver_adapter ALIAS kwiver_adapter )

viame_add_plugin( kwiver_processes_adapter
  KIND        PROCESSES
  SOURCES         register_processes.cxx
  PRIVATE         kwiver_adapter sprokit_pipeline
                  vital_config
)

viame_add_plugin( kwiver_epx_test
  KIND        PROCESSES
  SOURCES         epx_test.h
                  epx_test.cxx
                  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_epx_test_export.h
  PUBLIC          sprokit_pipeline
  PRIVATE         kwiver_adapter
                  vital_config
)

viame_lite_generated( kwiver_adapter_export.h pipeline_framework/adapters )
viame_lite_generated( kwiver_epx_test_export.h pipeline_framework/adapters )
viame_lite_generated( kwiver_processes_adapter_export.h pipeline_framework/adapters )
