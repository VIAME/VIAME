###
# GUI Project File Examples
##

set( example_files
  example_output.csv
  input_list.txt )

if( WIN32 )
  set( example_files
       ${example_files}
       run_chip_pipe.bat
       run_display_pipe.bat
       run_draw_pipe.bat )
else()
  set( example_files
       ${example_files}
       run_chip_pipe.sh
       run_display_pipe.sh
       run_draw_pipe.sh )
endif()

if( VIAME_ENABLE_VIVIA )
  set( example_files
    ${example_files}
    example_vpview.prj )

  if( WIN32 )
    set( example_files
         ${example_files}
         launch_view_interface.bat )
  elseif( APPLE )
    set( example_files
         ${example_files}
         launch_view_interface.mac.sh )
  else()
    set( example_files
         ${example_files}
         launch_view_interface.sh )
  endif()
endif()

if( VIAME_ENABLE_DIVE )
  if( WIN32 )
    set( example_files
         ${example_files}
         launch_dive_interface.bat )
  else()
    set( example_files
         ${example_files}
         launch_dive_interface.sh )
  endif()
endif()

if( VIAME_ENABLE_SEAL_TK )
  if( WIN32 )
    set( example_files
         ${example_files}
         launch_seal_interface.bat )
  elseif( APPLE )
    set( example_files
         ${example_files}
         launch_seal_interface.mac.sh )
  else()
    set( example_files
         ${example_files}
         launch_seal_interface.sh )
  endif()
endif()

install( FILES       ${example_files}
         COMPONENT VIAME
         DESTINATION examples/annotation_and_visualization )