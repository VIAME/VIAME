###
# Add matlab source files
##

set( plugin_matlab_files
  ScallopFinder.m
  ScallopFinder_proc.m
  check_configuration.m
  detect.m
  detector_initialize.m
  mahalanobis.m
  detect.m
  contourchains.m
  mahalanobis_mex.mexmaci64
  contourchains_mex.mexmaci64
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/lanl_scallop_finder )
