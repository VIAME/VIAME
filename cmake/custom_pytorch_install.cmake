message( "Running pytorch auxiliary install" )

if( WIN32 )
  set( DIR1 ${VIAME_PYTHON_BASE}/site-packages/torch/ )
  set( DIR2 ${VIAME_PYTHON_BASE}/dist-packages/torch/ )

  set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-compatibility-win32/include )

  if( EXISTS ${DIR1} )
    file( COPY ${PATCH} DESTINATION ${DIR1} )
  endif()

  if( EXISTS ${DIR2} )
    file( COPY ${PATCH} DESTINATION ${DIR2} )
  endif()
endif()

message( "Done" )
