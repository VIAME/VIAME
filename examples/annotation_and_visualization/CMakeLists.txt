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
       run_annotation_gui.bat
       run_chip_pipe.bat
       run_display_pipe.bat
       run_draw_pipe.bat )
elseif( APPLE )
  set( example_files
       ${example_files}
       run_annotation_gui.mac.sh
       run_chip_pipe.sh
       run_display_pipe.sh
       run_draw_pipe.sh )
else()
  set( example_files
       ${example_files}
       run_annotation_gui.sh
       run_chip_pipe.sh
       run_display_pipe.sh
       run_draw_pipe.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/annotation_and_visualization )
