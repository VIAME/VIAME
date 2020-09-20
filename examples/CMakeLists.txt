###
# VIAME examples
##

if( VIAME_INSTALL_EXAMPLES )

  add_subdirectory( annotation_and_visualization )
  add_subdirectory( archive_summarization )
  add_subdirectory( detection_file_conversions )
  add_subdirectory( example_imagery )
  add_subdirectory( external_plugin_creation )
  add_subdirectory( frame_level_classification )
  add_subdirectory( hello_world_pipeline )
  add_subdirectory( image_enhancement )
  add_subdirectory( image_registration )
  add_subdirectory( measurement_and_classification )
  add_subdirectory( measurement_using_stereo )
  add_subdirectory( object_detection )
  add_subdirectory( object_detector_training )
  add_subdirectory( object_tracking )
  add_subdirectory( object_tracker_training )
  add_subdirectory( pixel_level_classification )
  add_subdirectory( scoring_and_roc_generation )
  add_subdirectory( search_and_rapid_model_generation )
  add_subdirectory( using_detectors_in_cxx_code )

endif()

# Install overview readme
install( FILES README.rst DESTINATION examples )
