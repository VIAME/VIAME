#
# KWIVER processes
#
project( kwiver_core_processes )

set( sources
  register_processes.cxx

  compute_homography_process.cxx
  detect_features_process.cxx
  draw_tracks_process.cxx
  extract_descriptors_process.cxx
  frame_list_process.cxx
  matcher_process.cxx
  read_descriptor_process.cxx
  stabilize_image_process.cxx
  video_input_process.cxx
)

set( private_headers
  compute_homography_process.h
  detect_features_process.h
  draw_tracks_process.h
  extract_descriptors_process.h
  frame_list_process.h
  matcher_process.h
  read_descriptor_process.h
  stabilize_image_process.h
  video_input_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( kwiver_core_processes
  SUBDIR          sprokit
  SOURCES       ${sources}
                ${private_headers}
  PRIVATE         sprokit_pipeline
                  algorithms_core
                  vital vital_apm vital_logger vital_config
                  ${Boost_SYSTEM_LIBRARY}
                  ${Boost_FILESYSTEM_LIBRARY}
)
