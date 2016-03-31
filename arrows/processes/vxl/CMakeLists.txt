#
# KWIVER processes
#
project( kwiver_processes )

set( sources
  register_processes.cxx

  kw_archive_writer_process.cxx
)

set( private_headers
  kw_archive_writer_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

set( needed_libs       )


kwiver_add_plugin( kwiver_vxl_processes
  SUBDIR          sprokit
  SOURCES       ${sources}
                ${private_headers}
  PRIVATE         sprokit_pipeline
                  algorithms_core algorithms_vxl
                  vital vital_apm vital_logger vital_config
                  vsl vul vnl_algo vnl_io vnl vil_io vil
)

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
