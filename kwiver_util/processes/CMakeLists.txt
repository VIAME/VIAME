#
# KWIVER processes
#
project( kwiver_processes )

set( sources
  register_processes.cxx
  kw_archive_writer_process.cxx
  frame_list_process.cxx
  stabilize_image_process.cxx
)

set( private_headers
  register_processes.h
  kw_archive_writer_process.h
  frame_list_process.h
  stabilize_image_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

set( maptk_libs      maptk_core maptk_vxl )
if (KWIVER_ENABLE_OPENCV)
  set (maptk_libs    ${maptk_libs} maptk_ocv)
endif()

kwiver_add_plugin( kwiver_processes
  SOURCES         ${sources}
                  ${private_headers}
  PRIVATE         sprokit_pipeline
                  ${maptk_libs}
                  vital vital_apm vital_logger vital_config
                  ${Boost_FILESYSTEM_LIBRARY}
                  ${Boost_SYSTEM_LIBRARY}
                  vsl vil vul vnl vnl_io vil_io
)

# these are for the submodules
add_dependencies( kwiver_processes
  vital_proj sprokit_proj maptk_proj VXL )

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
