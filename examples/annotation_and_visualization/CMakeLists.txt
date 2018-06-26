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
       run_vpview.bat
       run_chip_pipe.bat
       run_display_pipe.bat
       run_draw_pipe.bat )
elseif( APPLE )
  set( example_files
       ${example_files}
       run_vpview.mac.sh
       run_chip_pipe.sh
       run_display_pipe.sh
       run_draw_pipe.sh )
else()
  set( example_files
       ${example_files}
       run_vpview.sh
       run_chip_pipe.sh
       run_display_pipe.sh
       run_draw_pipe.sh )

install( FILES       ${example_files}
         DESTINATION examples/annotation_and_visualization )
