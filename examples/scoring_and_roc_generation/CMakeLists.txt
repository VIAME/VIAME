###
# Scoring and ROC generation examples
##

set( example_files
  detections.csv
  groundtruth.csv )

if( VIAME_ENABLE_KWANT )
  if( WIN32 )
    set( example_files
         ${example_files}
         run_scoring_jointly_across_all.bat
         run_scoring_per_category.bat )
  else()
    set( example_files
         ${example_files}
         run_scoring_jointly_across_all.sh
         run_scoring_per_category.sh )
  endif()
endif()

install( FILES       ${example_files}
         DESTINATION examples/scoring_and_roc_generation )
