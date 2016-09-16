###
# Add matlab source files
##

set( plugin_matlab_files
  faster_rcnn_detector.h
  )

install( FILES ${plugin_matlab_files} DESTINATION matlab/matlab_ex )
