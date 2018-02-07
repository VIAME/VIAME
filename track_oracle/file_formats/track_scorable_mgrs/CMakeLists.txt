
#
# the scorable MGRS schema.  Doesn't really exist as a separate track
# file, but included here to simplify the template instances.
#

##
## The scorable_mgrs object
##

set( track_oracle_scorable_mgrs_public_headers
  scorable_mgrs.h
)

set( track_oracle_scorable_mgrs_sources
  scorable_mgrs.cxx
)

kwiver_install_headers(
  ${track_oracle_scorable_mgrs_public_headers}
  SUBDIR       track_oracle/file_formats/scorable_mgrs
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/scorable_mgrs_export.h
  NOPATH SUBDIR       track_oracle/file_formats/scorable_mgrs
)

kwiver_add_library( scorable_mgrs
  ${track_oracle_scorable_mgrs_public_headers}
  ${track_oracle_scorable_mgrs_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/scorable_mgrs_export.h
)

target_link_libraries( scorable_mgrs
  PUBLIC          kwiver_algo_proj
  PRIVATE         vital_logger
)

##
## The corresponding data term
##


set( track_oracle_scorable_mgrs_data_term_public_headers
  scorable_mgrs_data_term.h
)

set( track_oracle_scorable_mgrs_data_term_sources
  scorable_mgrs_data_term.cxx
  scorable_mgrs_data_term_instance.cxx
)

kwiver_install_headers(
  ${track_oracle_scorable_mgrs_data_term_public_headers}
  SUBDIR       track_oracle/scorable_mgrs
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/scorable_mgrs_data_term_export.h
  NOPATH SUBDIR       track_oracle/scorable_mgrs
)

kwiver_add_library( scorable_mgrs_data_term
  ${track_oracle_scorable_mgrs_public_headers}
  ${track_oracle_scorable_mgrs_data_term_public_headers}
  ${track_oracle_scorable_mgrs_data_term_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/scorable_mgrs_data_term_export.h
)

target_link_libraries( scorable_mgrs_data_term
  PUBLIC         track_oracle scorable_mgrs
  PRIVATE        vital_logger ${TinyXML_LIBRARY}
)

##
## the track object
##

set( track_oracle_track_scorable_mgrs_public_headers
  track_scorable_mgrs.h
)

set( track_oracle_track_scorable_mgrs_sources
  track_scorable_mgrs.cxx
)

kwiver_install_headers(
  ${track_oracle_track_scorable_mgrs_public_headers}
  SUBDIR       track_oracle/scorable_mgrs
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_scorable_mgrs_export.h
  NOPATH SUBDIR       track_oracle/scorable_mgrs
)

kwiver_add_library( track_scorable_mgrs
  ${track_oracle_track_scorable_mgrs_public_headers}
  ${track_oracle_track_scorable_mgrs_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_scorable_mgrs_export.h
)

if (KWANT_ENABLE_SHAPELIB)
    set(TRACK_APIX_LIB track_apix)
endif()

target_link_libraries( track_scorable_mgrs
  PUBLIC            track_oracle scorable_mgrs scorable_mgrs_data_term
  PRIVATE           track_kw18 vital_logger ${TRACK_APIX_LIB}
)
