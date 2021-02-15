set(CMAKE_FOLDER "Track Oracle")

OPTION(KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER  "Enable track_oracle's event_adapter (disable until further notice)" OFF )
OPTION(KWIVER_ENABLE_TRACK_ORACLE_MGRS  "Enable track_oracle's lat/lon <-> MGRS capabilities (disable until further notice)" OFF )

if (KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER)
  message( FATAL_ERROR "Building the event_adapter is blocked until VIBRANT lands" )
endif()
if (KWIVER_ENABLE_TRACK_MGRS)
  message( FATAL_ERROR "Building the MGRS code is blocked until lat/lon <-> MGRS capabilities are available" )
  set(TRACK_ORACLE_SCORABLE_MGRS_LIBRARY "scorable_mgrs")
  set(TRACK_ORACLE_SCORABLE_MGRS_DATA_TERM "scorable_mgrs_data_term")
else()
  set(TRACK_ORACLE_SCORABLE_MGRS_LIBRARY "" )
endif()

add_subdirectory( aries_interface )
add_subdirectory( utils )

##
## Core track_oracle
##

add_subdirectory( core )
add_subdirectory( vibrant_descriptors )
add_subdirectory( data_terms )
add_subdirectory( example )

##
## file formats
##

add_subdirectory( file_formats )

##
## Can't build the event_adapter until we have vidtk event types
## via vibrant
##

if (KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER)

set( track_oracle_event_adapter_public_headers
  event_adapter.h
)

set( track_oracle_event_adapter_sources
  event_adapter.cxx
)

kwiver_install_headers(
  ${track_oracle_event_adapter_public_headers}
  SUBDIR            track_oracle
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_event_adapter_export.h
  NOPATH SUBDIR            track_oracle
)

kwiver_add_library( track_oracle_event_adapter
  ${track_oracle_event_adapter_public_headers}
  ${track_oracle_event_adapter_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_event_adapter_export.h
)

target_link_libraries( track_oracle_event_adapter
  PUBLIC               logging_map
                       data_terms
  PRIVATE              track_oracle
                       scoring_aries_interface
                       vital_logger
                       vul
)

endif()

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
