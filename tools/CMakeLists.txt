###
# This file contains an assortment of VIAME standalone tools
##

###
# Python standalone tools
##
set( PYTHON_SCRIPTS
  add_segmentations.py
  convert_cam_format.py
  convert_itk_h5_transform.py
  database_tool.py
  extract_video_frames.py
  generate_detection_plots.py
  generate_nn_index.py
  launch_annotation_interface.py
  launch_search_interface.py
  plot_eval_results.py
  process_video.py
  survey_metadata.py
  train_detection_fusion.py
  viame_csv_tool.py
)

if( VIAME_ENABLE_OPENCV )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    calibrate_cameras.py
    compute_depth.py
    compute_disparity.py
    create_mosaic.py
    detect_prior_coverage.py
    ply_utilities.py
    reconstruct_3d.py
    stereo_rectify.py
  )
endif()

if( VIAME_ENABLE_PYTORCH )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    check_gpu_usability.py
  )
endif()

###
# C/C++ compiled standalone tools
##

# VIAME tool runner - main entry point similar to kwiver
kwiver_add_executable( viame
    viame.cxx
  )

target_link_libraries( viame
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::vital_applets
  )

###
# VIAME tool applets plugin - train and other applets
##
set( applet_sources
  register_applets.cxx
  csv.cxx
  get_configs.cxx
  resample_tracks.cxx
  score_results.cxx
  train.cxx
)

set( applet_headers
  applet_attributes.h
  csv.h
  get_configs.h
  resample_tracks.h
  score_results.h
  train.h
)

if( VIAME_ENABLE_PYTHON )
  set( applet_sources ${applet_sources} python_script_applet.cxx )
  set( applet_headers ${applet_headers} python_script_applet.h )
endif()

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( viame_tools_applets
  SUBDIR       viame/applets
  SOURCES      ${applet_sources} ${applet_headers}
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::vital_applets
               kwiver::sprokit_pipeline
               kwiver::kwiver_adapter
               viame_core
               viame_claude
)

if( VIAME_ENABLE_PYTHON )
  target_compile_definitions( viame_tools_applets
    PRIVATE VIAME_TOOLS_ENABLE_PYTHON )
endif()

# Only the plot image rendering needs OpenCV; the plot data itself is core
if( VIAME_ENABLE_OPENCV )
  target_compile_definitions( viame_tools_applets
    PRIVATE VIAME_TOOLS_HAVE_OPENCV )
  target_link_libraries( viame_tools_applets
    PRIVATE viame_opencv )
endif()

if( VIAME_ENABLE_PYTORCH )
  target_compile_definitions( viame_tools_applets
    PRIVATE VIAME_TOOLS_HAVE_PYTORCH )
endif()

if( VIAME_ENABLE_PYTHON )
  install( FILES       ${PYTHON_SCRIPTS}
           DESTINATION configs )
endif()

