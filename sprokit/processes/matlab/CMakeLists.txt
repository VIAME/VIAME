#
# KWIVER processes that use matlab
#
project( kwiver_matlab_processes )

set( sources
  register_processes.cxx

  matlab_process.cxx
)

set( private_headers
  matlab_process.h
)


kwiver_private_header_group( ${private_headers} )

include_directories( "${CMAKE_CURRENT_BINARY_DIR}" )
include_directories( "${Matlab_INCLUDE_DIRS}" )

kwiver_add_plugin( kwiver_processes_matlab
  SUBDIR           ${kwiver_plugin_process_subdir}
  SOURCES          ${sources}
                   ${private_headers}
  PRIVATE          sprokit_pipeline
                   kwiver_algo_core
                   kwiver_algo_matlab
                   kwiversys
                   vital vital_vpm vital_logger vital_config
                   ${Boost_SYSTEM_LIBRARY}
                   ${Boost_FILESYSTEM_LIBRARY}
                   ${Matlab_LIBRARIES}
)
