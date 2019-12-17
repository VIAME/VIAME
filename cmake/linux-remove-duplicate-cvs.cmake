
if( NOT WIN32 AND VIAME_ENABLE_OPENCV AND VIAME_ENABLE_PYTHON )

  file( GLOB BUILT_CV2 "${viame_python_install_path}/site-packages/cv2*.so" )
  file( GLOB PIP_CV2_HDR "${viame_python_install_path}/site-packages/opencv_python*" )

  set( PIP_CV2_LIB "${viame_python_install_path}/site-packages/cv2" )

  if( EXISTS "${BUILT_CV2}" AND EXISTS "${PIP_CV2_LIB}" AND EXISTS "${PIP_CV2_HDR}" )
    file( REMOVE_RECURSE ${PIP_CV2_LIB} )
    file( REMOVE_RECURSE ${PIP_CV2_HDR} )
  endif()
endif()
