#
# KWIVER processes
#
project( kwiver_processes )

# We are including sprokit macros because we are building
# sprokit plug-ins.
include("${sprokit_source_dir}/conf/sprokit-macro-targets.cmake")

set( sources
  register_processes.cxx
  kw_archive_writer_process.cxx
  frame_list_process.cxx
  stabilize_image_process.cxx
)

set( private_headers
  processes-config.h
  register_processes.h
  kw_archive_writer_process.h
  frame_list_process.h
  stabilize_image_process.h
)

sprokit_private_header_group( ${private_headers} )

# need sprokit macro to correctly process the plugin
sprokit_add_plugin( kwiver_processes
  MAKE_KWIVER_PROCESSES_LIB
  ${sources}
  ${private_headers}
)

## this may be done by the magic macros
# set_target_properties( kwiver_processes PROPERTIES VERSION ${KWIVER_VERSION} SOVERSION ${KWIVER_VERSION})

add_dependencies( kwiver_processes
  kwiver sprokit maptk VXL )

set( maptk_libs      maptk_core maptk_vxl )
if (KWIVER_ENABLE_OPENCV)
  set (maptk_libs    ${maptk_libs} maptk_ocv)
endif()

# depends on VXL sprokit maptk
target_link_libraries( kwiver_processes
  PRIVATE              sprokit_pipeline
                     ${maptk_libs}
                       kwiver
                     ${Boost_FILESYSTEM_LIBRARY}
                     ${Boost_SYSTEM_LIBRARY}
                       vsl vil vul vnl vnl_io vil_io )

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
