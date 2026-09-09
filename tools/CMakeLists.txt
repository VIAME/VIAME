###
# This file contains an assortment of VIAME standalone tools
##

###
# Python standalone tools
##
set( PYTHON_SCRIPTS
  add_segmentations.py
  convert_cam.py
  convert_itk.py
  database.py
  extract_frames.py
  plot_detections.py
  generate_nn_index.py
  launch_annotator.py
  launch_search.py
  pipeline.py
  plot_eval.py
  run_bulk.py
  metadata.py
  train_fusion.py
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
    rectify.py
  )
endif()

if( VIAME_ENABLE_PYTORCH )
  set( PYTHON_SCRIPTS
    ${PYTHON_SCRIPTS}
    check_gpu.py
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

# Only the applet attribute names are needed from core; linking it would load
# its every dependency on each viame invocation
target_include_directories( viame
  PRIVATE      ${VIAME_SOURCE_DIR}/plugins/core
  )

###
# VIAME tool applets plugin - train and other applets
##
set( applet_sources
  register_applets.cxx
  csv.cxx
  get_configs.cxx
  json.cxx
  resample.cxx
  score.cxx
  train.cxx
)

set( applet_headers
  csv.h
  get_configs.h
  json.h
  resample.h
  score.h
  train.h
)

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

# The json applet parses with the rapidjson vendored inside KWIVER's cereal,
# whose headers include each other relative to this directory
if( NOT KWIVER_SOURCE_DIR )
  set( KWIVER_SOURCE_DIR "${VIAME_PACKAGES_DIR}/kwiver" )
endif()
target_include_directories( viame_tools_applets
  PRIVATE      ${KWIVER_SOURCE_DIR}/vital/internal
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

