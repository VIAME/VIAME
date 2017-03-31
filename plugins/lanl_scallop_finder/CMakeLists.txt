###
# Add matlab source files
##

set( plugin_matlab_files
  ScallopFinder.m
  check_configuration.m
  detector_initialize.m
  mahalanobis_mex.mexmaci64
  contourchains_mex.mexmaci64
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/lanl_scallop_finder )
