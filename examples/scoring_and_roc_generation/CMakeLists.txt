###
# Scoring and ROC generation examples
##

set( example_files
  plotroc.py
  detections.csv
  detections.kw18
  groundtruth.csv
  groundtruth.kw18 )

if( VIAME_ENABLE_KWANT )
  if( WIN32 )
    set( example_files
         ${example_files}
         run_scoring.bat )
  else()
    set( example_files
         ${example_files}
         run_scoring.sh )
  endif()
endif()

install( FILES       ${example_files}
         DESTINATION examples/scoring_and_roc_generation )
