#
# KWIVER processes
#
project( kwiver_processes )

## include( GenerateExportHeader )

# We are including sprokit macros because we are building
# sprokit plug-ins.
include("${sprokit_source_dir}/conf/sprokit-macro-targets.cmake")

# or include( "${sprokit_build_dir}/sprokit-macro-targets.cmake")

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

sprokit_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

# need sprokit macro to correctly process the plugin
kwiver_add_plugin( kwiver_processes
  ${sources}
  ${private_headers}
)

add_dependencies( kwiver_processes
  vital_proj sprokit_proj maptk_proj VXL )

set( maptk_libs      maptk_core maptk_vxl )
if (KWIVER_ENABLE_OPENCV)
  set (maptk_libs    ${maptk_libs} maptk_ocv)
endif()

# depends on VXL sprokit maptk
target_link_libraries( kwiver_processes
  PRIVATE              sprokit_pipeline
                     ${maptk_libs}
                     vital vital_apm vital_logger vital_config
                     ${Boost_FILESYSTEM_LIBRARY}
                     ${Boost_SYSTEM_LIBRARY}
                       vsl vil vul vnl vnl_io vil_io )

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()

###
# Make appropriate export header file
## generate_export_header( kwiver_processes )
