#
# Video metadata processing library
#

set( sources
  video_metadata_util.cxx
  convert_metadata.cxx
  convert_0601_metadata.cxx
  convert_0104_metadata.cxx
  )

set( public_headers
  video_metadata_util.h
  convert_metadata.h
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
