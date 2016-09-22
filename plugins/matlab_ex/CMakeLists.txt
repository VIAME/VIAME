###
# Add matlab source files
##

set( plugin_matlab_files
  hello_world_matlab_detector.m
  detect.m
  detector_initialize.m
  check_configuration.m
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/matlab_ex )
