# Scripts are the one thing in examples that only ever comes from this source
# tree, so clear any previously installed ones before the install rules put
# back the current set. A renamed or deleted example script otherwise stays in
# the install forever, and the example tests enumerate installed scripts -- a
# leftover becomes a test for an example that no longer exists, failing against
# configs that are no longer shipped (train_netharn_cfrnn_habcam_csv.sh was
# doing exactly that). Data directories and generated outputs that live under
# examples/ are untouched.
install( CODE "
  file( GLOB_RECURSE _viame_stale_scripts
        \"\${CMAKE_INSTALL_PREFIX}/examples/*.sh\"
        \"\${CMAKE_INSTALL_PREFIX}/examples/*.bat\" )
  list( LENGTH _viame_stale_scripts _viame_stale_count )
  if( _viame_stale_count GREATER 0 )
    message( STATUS
      \"Clearing \${_viame_stale_count} previously installed example script(s)\" )
    file( REMOVE \${_viame_stale_scripts} )
  endif()
" )

###
# VIAME examples
##

if( VIAME_INSTALL_EXAMPLES )

  add_subdirectory( annotation_and_visualization )
  add_subdirectory( archive_summarization )
  add_subdirectory( detection_file_conversions )
  add_subdirectory( example_imagery )
  add_subdirectory( frame_level_classification )
  add_subdirectory( example_pipeline )
  add_subdirectory( image_enhancement )
  add_subdirectory( object_detection )
  add_subdirectory( object_detector_training )
  add_subdirectory( object_tracking )
  add_subdirectory( object_tracker_training )
  add_subdirectory( plugin_creation )
  add_subdirectory( registration_and_mosaicing )
  add_subdirectory( search_and_rapid_model_generation )
  add_subdirectory( scene_segmentation )
  add_subdirectory( scoring_and_evaluation )
  add_subdirectory( size_measurement )
  add_subdirectory( using_algorithms_in_code )

endif()

# Install overview readme
install( FILES README.rst DESTINATION examples )
