#
# KWIVER adapter processes
#
project( kwiver_adapter_processes )

set( proc_sources
  register_processes.cxx

  adapter_data_set.cxx   adapter_data_set.h
  adapter_base.cxx
  input_adapter_process.cxx
  output_adapter_process.cxx
  )

set( private_headers
  adapter_base.h
  input_adapter_process.h
  output_adapter_process.h
  )

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

###
# make processes plugin
kwiver_add_plugin( kwiver_processes_adapter
  SUBDIR          sprokit
  SOURCES         ${proc_sources}
                  ${private_headers}
  PRIVATE         sprokit_pipeline
                  vital vital_vpm vital_logger vital_config
)

###
# library components
set( lib_sources
  adapter_data_set.cxx
  input_adapter.cxx
  output_adapter.cxx

  embedded_pipeline.cxx

  adapter_base.h             adapter_base.cxx
  input_adapter_process.h
  output_adapter_process.h
  )

set( public_headers
  adapter_types.h
  adapter_data_set.h
  input_adapter.h
  output_adapter.h
  embedded_pipeline.h
  )

###
# Make external interface library
kwiver_add_library( kwiver_adapter
  ${lib_sources}
  ${public_headers}
  )

target_link_libraries( kwiver_adapter
  PUBLIC           vital
                   vital_vpm
                   vital_logger
  PRIVATE          sprokit_pipeline
                   sprokit_tools

  )

kwiver_install_headers( ${public_headers}
  SUBDIR sprokit/processes/adapters
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_adapter_export.h
  SUBDIR     sprokit/processes/adapters
  NOPATH
)

if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
