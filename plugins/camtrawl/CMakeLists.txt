###
# Add matlab source files
##

set( plugin_matlab_files
  initialize_gmm.m
  gmm_background_remove.m
  extract_targets2.m
  find_match.m
  imOrientedBoxx.m
  normalize_pixel.m
  project_points2.m
  rigid_motion.m
  rodrigues.m
  stereo_triangulation.m
  detect.m
  detector_initialize.m
  check_configuration.m
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/matlab_ex )
