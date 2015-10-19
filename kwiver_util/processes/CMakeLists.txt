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
  find_package( OpenCV REQUIRED )
  if( OpenCV_VERSION VERSION_LESS "2.4" )
    message( FATAL_ERROR "OpenCV version must be at least 2.4" )
  endif()

  set (maptk_libs    ${maptk_libs}
    maptk_ocv)

  kwiver_add_plugin( kwiver_processes_ocv
    SUBDIR           sprokit
    SOURCES          register_processes_ocv.cxx
                     view_image_process.cxx     view_image_process.h
    PRIVATE          sprokit_pipeline
                     vital vital_apm vital_logger vital_config
                     vsl vul vnl_algo vnl_io vnl vil_io vil
                     ${maptk_libs}
                     ${OpenCV_LIBS}
   )

endif()

kwiver_add_plugin( kwiver_processes
  SUBDIR          sprokit
  SOURCES       ${sources}
                ${private_headers}
  PRIVATE         sprokit_pipeline
                ${maptk_libs}
                  vital vital_apm vital_logger vital_config
                ${ocv_libs}
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
