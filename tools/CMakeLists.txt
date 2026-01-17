###
# This file contains an assortment of VIAME standalone tools
##

###
# Python standalone tools
##
set( PYTHON_SCRIPTS
  convert_cam_format.py
  database_tool.py
  extract_video_frames.py
  generate_detection_plots.py
  generate_nn_index.py
  launch_annotation_interface.py
  launch_search_interface.py
  launch_timeline_interface.py
  plot_eval_results.py
  process_video.py
  score_results.py
  viame_csv_tool.py
)

if( VIAME_ENABLE_OPENCV )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    calibrate_cameras.py
    compute_depth.py
    compute_disparity.py
    create_mosaic.py
    ply_utilities.py
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
               kwiver::kwiver_tools_applet
  )

###
# VIAME tool applets plugin - train and other applets
##
set( applet_sources
  register_applets.cxx
  train.cxx
)

set( applet_headers
  train.h
)

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( viame_tools_applets
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${applet_sources} ${applet_headers}
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::kwiver_tools_applet
               kwiver::sprokit_pipeline
               kwiver::kwiver_adapter
               viame_core
)

kwiver_add_executable( viame_get_configs
    get_configs.cxx
  )

target_link_libraries( viame_get_configs
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::kwiversys
               kwiver::sprokit_pipeline
  )

if( VIAME_ENABLE_OPENCV )
  kwiver_add_executable( viame_score_results
    score_results.cxx
  )

  target_link_libraries( viame_score_results
    PRIVATE      kwiver::vital
                 kwiver::vital_vpm
                 kwiver::vital_logger
                 kwiver::kwiversys
                 viame_core
                 viame_opencv
    )
endif()

if( VIAME_ENABLE_PYTHON )
  install( FILES       ${PYTHON_SCRIPTS}
           DESTINATION configs )
endif()

