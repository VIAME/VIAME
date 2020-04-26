###
# This file contains an assortment of VIAME standalone tools
##

###
# Python standalone tools
##
set( PYTHON_SCRIPTS
  check_formatting.py
  database_tool.py
  extract_video_frames.py
  process_video.py
  launch_search_interface.py
  launch_timeline_interface.py
  launch_annotation_interface.py
  generate_detection_plots.py
  score_results.py )

###
# C/C++ compiled standalone tools
##
kwiver_add_executable( viame_train_detector
  viame_train_detector.cxx
  )

find_package( Boost ${KWIVER_BOOST_VERSION} REQUIRED
    COMPONENTS
      chrono
      filesystem
      system)

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

target_link_libraries( viame_train_detector
  PRIVATE      kwiver::vital
               kwiver::vital_vpm
               kwiver::vital_config
               kwiver::vital_exceptions
               kwiver::vital_logger
               kwiver::vital_algo
               kwiver::vital_util
               kwiver::kwiversys
               kwiver::sprokit_pipeline
               kwiver::kwiver_adapter
               ${Boost_SYSTEM_LIBRARY}
               ${Boost_FILESYSTEM_LIBRARY}
  )

if( VIAME_ENABLE_PYTHON )
  install( FILES       ${RUN_SCRIPT_FILES}
           DESTINATION configs )
endif()
