#
# KWIVER processes that use OpenCV
#
project( kwiver_ocv_processes )

set( sources
  register_processes.cxx

  image_viewer_process.cxx
  detect_in_subregions_process.cxx
)

set( private_headers
  image_viewer_process.h
  detect_in_subregions_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

find_package( OpenCV REQUIRED )
if( OpenCV_VERSION VERSION_LESS "2.4" )
  message( FATAL_ERROR "OpenCV version must be at least 2.4" )
endif()

kwiver_add_plugin( kwiver_processes_ocv
  SUBDIR           ${kwiver_plugin_process_subdir}
  SOURCES          ${sources}
                   ${private_headers}
  PRIVATE          sprokit_pipeline
                   kwiver_algo_core kwiver_algo_ocv
                   vital vital_vpm vital_logger vital_config
                   kwiversys
                   ${OpenCV_LIBS}
 )
