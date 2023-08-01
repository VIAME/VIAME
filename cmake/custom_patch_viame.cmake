# Move any misinstalled python files
if( PYTHON_VERSION_STRING )
  # Sometimes VIAME subpackages install python files to incorrect python
  # subdirectories, like lib/site-packages instead of lib/pythonX.Y/site-packages
  set( ROOT_PYTHON_DIR "${VIAME_INSTALL_PREFIX}/lib/${PYTHON_VERSION_STRING}" )
  set( OUTPUT_PYTHON_DIR "${ROOT_PYTHON_DIR}/site-packages/" )
  set( PYTHON_VERSION_APPENDED "${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}" )

  if( EXISTS ${VIAME_INSTALL_PREFIX}/Python${PYTHON_VERSION_APPENDED} )
    message( WARNING "Relocating misinstalled ${VIAME_INSTALL_PREFIX}/Python${PYTHON_VERSION_APPENDED}" )
    file( GLOB FILES_TO_MOVE "${VIAME_INSTALL_PREFIX}/Python${PYTHON_VERSION_APPENDED}/site-packages/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE "${VIAME_INSTALL_PREFIX}/Python${PYTHON_VERSION_APPENDED}" )
  endif()

  if( EXISTS ${VIAME_INSTALL_PREFIX}/lib/site-packages )
    message( WARNING "Relocating misinstalled ${VIAME_INSTALL_PREFIX}/lib/site-packages" )
    file( GLOB FILES_TO_MOVE "${VIAME_INSTALL_PREFIX}/lib/site-packages/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE "${VIAME_INSTALL_PREFIX}/lib/site-packages" )
  endif()
endif()