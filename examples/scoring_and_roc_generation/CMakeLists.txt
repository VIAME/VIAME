###
# Scoring and ROC generation examples
##

set( example_files
  plotroc.py
  detections.kw18
  groundtruth.kw18
  README.md )

if( WIN32 )
  set( example_files
       ${example_files}
       run_scoring.bat )
else()
  set( example_files
       ${example_files}
       run_scoring.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/scoring_and_roc_generation )
