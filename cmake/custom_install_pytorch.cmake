message( "Running pytorch auxiliary install" )

if( WIN32 )

  # Patches directly to pytorch
  set( TORCH_DIR ${VIAME_PYTHON_BASE}/site-packages/torch )

  if( NOT EXISTS ${TORCH_DIR} )
    set( TORCH_DIR ${VIAME_PYTHON_BASE}/dist-packages/torch )
  endif()

  if( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.7.1" )
    set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-win32-1.7.1/include )
  elseif( VIAME_PYTORCH_VERSION VERSION_EQUAL "1.4.0" )
    set( PATCH ${VIAME_PATCH_DIR}/pytorch/mmcv-win32-1.4.0/include )
  endif()

  if( EXISTS ${TORCH_DIR} )
    file( COPY ${PATCH}
          DESTINATION ${TORCH_DIR} )
  endif()

endif()

message( "Done" )
