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
       run_vpview.bat )
elseif( APPLE )
  set( example_files
       ${example_files}
       run_vpview.mac.sh )
else()
  set( example_files
       ${example_files}
       run_vpview.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/annotation_and_visualization )
