#
# Video metadata processing library
#

set( sources
  video_metadata.cxx
  video_metadata_traits.cxx
  video_metadata_util.cxx
  convert_metadata.cxx
  convert_0601_metadata.cxx
  convert_0104_metadata.cxx
  pos_metadata_io.cxx
  )

set( public_headers
  video_metadata.h
  video_metadata_map.h
  video_metadata_traits.h
  video_metadata_tags.h
  video_metadata_util.h
  convert_metadata.h
  pos_metadata_io.h
  )

set( private_headers
  )

kwiver_add_library( vital_video_metadata
  ${sources}
  ${public_headers}
  ${private_headers}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_video_metadata_export.h
  )

target_link_libraries( vital_video_metadata
  PRIVATE              kwiversys
                       vital
                       vital_klv
                       vital_logger
)

kwiver_install_headers(
    ${public_headers}
    SUBDIR     vital/video_metadata
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_video_metadata_export.h
  NOPATH
  SUBDIR      vital/video_metadata
  )

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
