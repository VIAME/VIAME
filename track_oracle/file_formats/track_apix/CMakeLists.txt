#
# APIX tracks
#

find_package(SHAPELIB REQUIRED)
include_directories( SYSTEM ${SHAPELIB_INCLUDE_DIR} )

find_package(GeographicLib REQUIRED)
include_directories(SYSTEM ${GeographicLib_INCLUDE_DIR})

set( track_apix_public_headers
  track_apix.h
  track_apix_writer.h
  file_format_apix.h
)

set( track_apix_sources
  track_apix_writer.cxx
  file_format_apix.cxx
)

kwiver_install_headers(
  ${track_apix_public_headers}
  SUBDIR          track_oracle/file_formats/track_apix
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_apix_export.h
  NOPATH SUBDIR          track_oracle/file_formats/track_apix
)

kwiver_add_library( track_apix
  ${track_apix_public_headers}
  ${track_apix_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_apix_export.h
)

target_link_libraries( track_apix
  PUBLIC               track_oracle
                       track_oracle_format_base
                       data_terms
  PRIVATE              kwiver_geographic
                       ${SHAPELIB_LIBRARY}
                       ${Boost_DATE_TIME_LIBRARY}
)


