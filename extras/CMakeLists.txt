###
# Extras

option( KWIVER_ENABLE_RightTrack "Add RightTrack integration" OFF )

# Add common include directories to be made available for extras
include_directories ( "${KWIVER_SOURCE_DIR}" )
include_directories ( "${KWIVER_SOURCE_DIR}/sprokit/src" )
include_directories ( "${KWIVER_BINARY_DIR}/sprokit/src" )

if (KWIVER_ENABLE_RightTrack)
  add_subdirectory(RightTrack)
endif()

add_subdirectory( process_instrumentation )
