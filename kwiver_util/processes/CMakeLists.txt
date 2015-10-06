#
# KWIVER processes
#
project( kwiver_processes )

set( sources
  register_processes.cxx

  frame_list_process.cxx
  stabilize_image_process.cxx
  kw_archive_writer_process.cxx

  detect_features_process.cxx
  extract_descriptors_process.cxx
  matcher_process.cxx
  compute_homography_process.cxx
)

set( private_headers
  register_processes.h

  frame_list_process.h
  stabilize_image_process.h
  kw_archive_writer_process.h

  detect_features_process.h
  extract_descriptors_process.h
  matcher_process.h
  compute_homography_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

set( maptk_libs      maptk_core maptk_vxl )
if (KWIVER_ENABLE_OPENCV)
  set (maptk_libs    ${maptk_libs} maptk_ocv)
endif()

# link_directories( /home/linus/projects/KWIVER/keith/build/install/lib )
kwiver_add_plugin( kwiver_processes
  SUBDIR          sprokit
  SOURCES       ${sources}
                ${private_headers}
  PRIVATE         sprokit_pipeline
                ${maptk_libs}
                  vital vital_apm vital_logger vital_config
                ${Boost_FILESYSTEM_LIBRARY}
                ${Boost_SYSTEM_LIBRARY}
                  vsl vul vnl_algo vnl_io vnl vil_io vil
)

# these are for the submodules
add_dependencies( kwiver_processes
  vital_proj sprokit_proj maptk_proj VXL )

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif()
