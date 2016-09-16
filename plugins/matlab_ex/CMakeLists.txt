###
# Add matlab source files
##

set( plugin_matlab_files
  simple_detector.m
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/matlab_ex )
