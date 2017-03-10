###
# VIAME examples
##

option( VIAME_INSTALL_EXAMPLES  "Install existing VIAME examples"     ON )
option( VIAME_DOWNLOAD_MODELS   "Download example detection models"  OFF )

if( VIAME_INSTALL_EXAMPLES )

  add_subdirectory( detection_file_conversions )
  add_subdirectory( detector_model_training )
  add_subdirectory( detector_pipelines )
  add_subdirectory( hello_world_pipeline )
  add_subdirectory( using_detectors_in_cxx_code )

endif()
