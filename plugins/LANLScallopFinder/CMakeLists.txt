###
# Add matlab source files
##

set( plugin_matlab_files
  ScallopFinder.m
  ScallopFinder_proc.m
  check_configuration.m
  detect.m
  detector_initialize.m
  mahalanobis_mex.mexmaci64
  contourchains_mex.mexmaci64
  )

install( FILES ${plugin_matlab_files} DESTINATION /Users/prasad/VIAME/VIAME-build/install/matlab/LANLScallopFinder )
