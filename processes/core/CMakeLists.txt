#
# KWIVER processes
#
project( kwiver_core_processes )

set( sources
  register_processes.cxx

  frame_list_process.cxx
  video_input_process.cxx
  stabilize_image_process.cxx

  detect_features_process.cxx
  extract_descriptors_process.cxx
  matcher_process.cxx
  compute_homography_process.cxx

  draw_tracks_process.cxx
  read_descriptor_process.cxx
)

set( private_headers
  frame_list_process.h
  video_input_process.h
  stabilize_image_process.h

  detect_features_process.h
  extract_descriptors_process.h
  matcher_process.h
  compute_homography_process.h

  draw_tracks_process.h
  read_descriptor_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( kwiver_core_processes
  SUBDIR          sprokit
  SOURCES       ${sources}
                ${private_headers}
  PRIVATE         sprokit_pipeline
                  maptk_core
                  vital vital_apm vital_logger vital_config
)
