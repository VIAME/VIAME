###
# GUI Project File Examples
##

set( example_files
  example_output.csv
  input_list.txt )

if( WIN32 )
  set( example_files
       ${example_files}
       draw_detections_on_frames.bat
       extract_chips_from_detections.bat
       extract_frames_with_tracks.bat
       simple_pipeline_display.bat )
else()
  set( example_files
       ${example_files}
       draw_detections_on_frames.sh
       extract_chips_from_detections.sh
       extract_frames_with_tracks.sh
       simple_pipeline_display.sh )
endif()

if( VIAME_ENABLE_VIVIA )
  set( example_files
    ${example_files}
    vpview_example_project.prj )

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

if( VIAME_ENABLE_SEAL )
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

install( FILES        ${example_files}
         DESTINATION  examples/annotation_and_visualization )
