###
# Application level pipelines
##
set( HELLO_WORLD_EX_SOURCE
     hello_world_detector.pipe
     input_list.txt
     )

if( WIN32 )
  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       run_example.bat
       )
else()
  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       run_example.sh
       )
endif()

if( VIAME_ENABLE_PYTHON )

  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       hello_world_python.pipe
       )

  if( WIN32 )
    set( HELLO_WORLD_EX_SOURCE
         ${HELLO_WORLD_EX_SOURCE}
         run_python_example.bat
         )
  else()
    set( HELLO_WORLD_EX_SOURCE
         ${HELLO_WORLD_EX_SOURCE}
         run_python_example.sh
         )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${HELLO_WORLD_EX_SOURCE}
         DESTINATION examples/example_pipeline )
