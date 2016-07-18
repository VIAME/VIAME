#
# KWIVER processes that use OpenCV
#
project( kwiver_ocv_processes )

set( sources
  register_processes.cxx

  draw_detected_object_boxes_process.cxx
  view_image_process.cxx
)

set( private_headers
  draw_detected_object_boxes_process.h
  view_image_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

find_package( OpenCV REQUIRED )
if( OpenCV_VERSION VERSION_LESS "2.4" )
  message( FATAL_ERROR "OpenCV version must be at least 2.4" )
endif()

kwiver_add_plugin( kwiver_processes_ocv
  SUBDIR           sprokit
  SOURCES        ${sources}
                 ${private_headers}
  PRIVATE          sprokit_pipeline
                   kwiver_algo kwiver_algo_ocv
                   vital vital_apm vital_logger vital_config
                 ${OpenCV_LIBS}
 )
