#
#
#
if ( KWIVER_ENABLE_VXL AND KWIVER_ENABLE_OPENCV )
  add_subdirectory( call_SMQTK_pipeline )
endif()

add_subdirectory( process_template )
