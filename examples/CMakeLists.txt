###
# VIAME examples
##

if( VIAME_INSTALL_EXAMPLES )

  add_subdirectory( common_data )
  add_subdirectory( detection_file_conversions )
  add_subdirectory( detector_training_old_api )
  add_subdirectory( detector_pipelines )
  add_subdirectory( external_plugin_creation )
  add_subdirectory( hello_world_pipeline )
  add_subdirectory( scoring_and_roc_generation )
  add_subdirectory( tracking_pipelines )
  add_subdirectory( using_detectors_in_cxx_code )
  add_subdirectory( visualizing_detections_in_gui )

endif()
