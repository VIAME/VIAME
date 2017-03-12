###
# GUI Project File Examples
##

set( example_files
  display_annotation.pipe
  example_detections.kw18
  example_tot_scores.txt
  example_vpview.prj
  input_list.txt )

if( WIN32 )
  set( example_files
       ${example_files}
       run_example.bat )
else()
  set( example_files
       ${example_files}
       run_example.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/visualizing_detections_in_gui )
