message("Running smqtk post-install cleanup")

include( ${VIAME_CMAKE_DIR}/common_macros.cmake )

# SVM install tree quick hacks
set( PYFROM "${VIAME_INSTALL_PREFIX}/lib/python3" )
set( PYTO "${VIAME_PYTHON_INSTALL}/site-packages" )

if( EXISTS "${PYFROM}" AND EXISTS "${PYTO}" )
  if( EXISTS "${PYFROM}/site-packages/svm.py" )
    file( COPY "${PYFROM}/site-packages/svm.py"
          DESTINATION "${PYTO}" )
    file( COPY "${PYFROM}/site-packages/svmutil.py"
          DESTINATION "${PYTO}" )
  elseif( EXISTS "${PYFROM}/dist-packages/svm.py" )
    file( COPY "${PYFROM}/dist-packages/svm.py"
          DESTINATION "${PYTO}" )
    file( COPY "${PYFROM}/dist-packages/svmutil.py"
          DESTINATION "${PYTO}" )
  endif()
  file( REMOVE_RECURSE "${PYFROM}" )
endif()

if( NOT WIN32 )
  CreateSymlink( "${VIAME_INSTALL_PREFIX}/lib/libsvm.so.2"
                 "${VIAME_INSTALL_PREFIX}/lib/libsvm.so" )
endif()

message("Done")
