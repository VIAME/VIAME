###
# Add matlab source files
##

set( plugin_matlab_files
  camtrawl_detector.m
  check_configuration.m
  detect.m
  detector_initialize.m
  extract_chip_coords2.m
  extract_targets2.m
  gmm_background_remove.m
  imOrientedBoxx.m
  initialize_gmm.m
)

install( FILES ${plugin_matlab_files} DESTINATION matlab/camtrawl )
