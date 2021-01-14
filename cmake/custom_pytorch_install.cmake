message( "Running pytorch auxiliary install" )

if( WIN32 )
  set( DIR1 ${VIAME_PYTHON_BASE}/site-packages/torch/ )
  set( DIR2 ${VIAME_PYTHON_BASE}/dist-packages/torch/ )

  if( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.7.1" )
    set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-win32-1.7.1/include )
  elseif( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.4.0" )
    set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-win32-1.4.0/include )
  endif()

  if( EXISTS ${DIR1} )
    file( COPY ${PATCH} DESTINATION ${DIR1} )
  endif()

  if( EXISTS ${DIR2} )
    file( COPY ${PATCH} DESTINATION ${DIR2} )
  endif()
endif()

message( "Done" )
