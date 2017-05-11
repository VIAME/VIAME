set( vibrant_descriptors_public_headers
  descriptor_cutic_type.h
  descriptor_event_label_type.h
  descriptor_metadata_type.h
  descriptor_motion_type.h
  descriptor_overlap_type.h
  descriptor_raw_1d_type.h
)

set( vibrant_descriptors_private_headers
  track_field_output_specializations.h
)

set( vibrant_descriptors_sources
  descriptor_cutic_type.cxx
  descriptor_event_label_type.cxx
  descriptor_metadata_type.cxx
  descriptor_motion_type.cxx
  descriptor_overlap_type.cxx
  descriptor_raw_1d_type.cxx
  vibrant_descriptor_instances.cxx
  track_field_output_specializations.cxx
)

kwiver_install_headers(
  ${vibrant_descriptors_public_headers}
  SUBDIR        track_oracle/vibrant_descriptors
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vibrant_descriptors_export.h
  NOPATH SUBDIR        track_oracle/vibrant_descriptors
)

kwiver_add_library( vibrant_descriptors
  ${vibrant_descriptors_public_headers}
  ${vibrant_descriptors_private_headers}
  ${vibrant_descriptors_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vibrant_descriptors_export.h
)

target_link_libraries( vibrant_descriptors
  PUBLIC               track_oracle
  PRIVATE              vital_logger
                       ${TinyXML_LIBRARY}
)