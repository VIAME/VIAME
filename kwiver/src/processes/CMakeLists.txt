#
# KWIVER processes
#

# Sprokit macros
include("${sprokit_source_dir}/conf/sprokit-macro-targets.cmake")

if (KWIVER_ENABLE_PYTHON)
  include("${sprokit_source_dir}/conf/sprokit-macro-configure.cmake")
  include("${sprokit_source_dir}/conf/sprokit-macro-python.cmake")
endif()


SET( sources
  register_processes.cxx
  kw_archive_writer_process.cxx
  frame_list_process.cxx
  stabilize_image_process.cxx
)

set( private_headers
  kwiver-config.h
  register_processes.h
  kw_archive_writer_process.h
  frame_list_process.h
  stabilize_image_process.h
)

sprokit_private_header_group(${private_headers})

# need sprokit macro to correctly process the plugin
sprokit_add_plugin( kwiver_processes
  MAKE_KWIVER_PROCESSES_LIB
  ${sources}
  ${private_headers}
)

# set_target_properties( kwiver_processes PROPERTIES VERSION ${KWIVER_VERSION} SOVERSION ${KWIVER_VERSION})

add_dependencies( kwiver_processes kwiver_core sprokit maptk VXL )

set( maptk_libs maptk_core maptk_vxl )
if (KWIVER_ENABLE_OPENCV)
  set (maptk_libs ${maptk_libs} maptk_ocv)
endif()

# depends on VXL sprokit maptk
target_link_libraries( kwiver_processes
  LINK_PRIVATE
    sprokit_pipeline
    ${maptk_libs}
    kwiver_core
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    vsl vil vul vnl vnl_io vil_io )

if (KWIVER_ENABLE_PYTHON)
  set(sprokit_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
  set(sprokit_python_output_path "${KWIVER_BUILD_INSTALL_PREFIX}/lib/${sprokit_python_subdir}")
  sprokit_create_python_plugin_init(kwiver)
  sprokit_create_python_plugin_init(kwiver/processes)
  sprokit_add_python_module(${CMAKE_SOURCE_DIR}/kwiver/src/processes/kw_print_number_process.py kwiver/processes kw_print_number_process)
endif()
