###
# Add matlab source files
##

set( plugin_matlab_files
  initialize_gmm.m
  gmm_background_remove.m
  extract_targets2.m
  extract_chip_coords.m
  imOrientedBoxx.m
  detect.m
  detector_initialize.m
  check_configuration.m
  camtrawl_detector.m
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/camtrawl )
