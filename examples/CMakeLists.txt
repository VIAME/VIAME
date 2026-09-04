# Installs never remove files, so clear stale example scripts before
# reinstalling; the example tests enumerate whatever is installed.
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
