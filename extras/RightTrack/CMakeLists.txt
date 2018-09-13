###
# RightTrack plugin

# Should use find module for this, but RightTrack needs to be upgraded.
find_library (RIGHT_TRACK_LIB right_track )
SET ( RIGHT_TRACK_DIR "" CACHE FILEPATH "Location of include files" )

include_directories ( "${RIGHT_TRACK_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" )

kwiver_add_plugin( RightTrack_plugin
  SOURCES          register_plugin.cxx
                   rt_process_instrumentation.h
                   rt_process_instrumentation.cxx
                   ${CMAKE_CURRENT_BINARY_DIR}/righttrack_plugin_export.h
  PRIVATE          vital
                   vital_vpm
                   "${RIGHT_TRACK_LIB}"
                   vcl vpl vul vsl
                   sprokit_pipeline
  SUBDIR           ${kwiver_plugin_process_instrumentation_subdir}
  )
