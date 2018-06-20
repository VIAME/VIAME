###
# GUI Project File Examples
##

set( example_files
  example_output.csv
  example_vpview.prj
  input_list.txt )

if( WIN32 )
  set( example_files
       ${example_files}
       run_vpview.bat )
elseif( APPLE )
  set( example_files
       ${example_files}
       run_vpview.mac.sh
       run_pipe_display.sh )
else()
  set( example_files
       ${example_files}
       run_vpview.sh
       run_pipe_display.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/annotation_and_visualization )
