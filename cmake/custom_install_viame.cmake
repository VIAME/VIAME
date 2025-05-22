
# Paths used across multiple checks
set( OUTPUT_BIN_DIR "${VIAME_INSTALL_PREFIX}/bin" )

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

  if( EXISTS ${OUTPUT_PYTHON_DIR}/win32/lib )
    message( WARNING "Relocating misinstalled ${OUTPUT_PYTHON_DIR}/win32/lib" )
    file( GLOB FILES_TO_MOVE ${OUTPUT_PYTHON_DIR}/win32/lib )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE ${OUTPUT_PYTHON_DIR}/win32/lib )
  endif()

  if( EXISTS ${OUTPUT_PYTHON_DIR}/win32 )
    message( WARNING "Relocating misinstalled ${OUTPUT_PYTHON_DIR}/win32" )
    file( GLOB FILES_TO_MOVE ${OUTPUT_PYTHON_DIR}/win32 )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE ${OUTPUT_PYTHON_DIR}/win32 )
  endif()
endif()

# Move any misinstalled darknet executables
if( VIAME_ENABLE_DARKNET )
  if( WIN32 )
    set( DARKNET_BIN_WIN "${VIAME_BUILD_PREFIX}/src/darknet-build/Release/darknet.exe" )
    set( PTHREADS_DLL_WIN "${VIAME_SOURCE_PREFIX}/packages/darknet/3rdparty/pthreads/bin/pthreadVC2.dll" )

    if( EXISTS "${DARKNET_BIN_WIN}" )
      message( WARNING "Relocating misinstalled darknet executable" )
      file( COPY ${DARKNET_BIN_WIN} DESTINATION ${OUTPUT_BIN_DIR} )
    endif()
    if( EXISTS ${PTHREADS_DLL_WIN} )
      message( WARNING "Relocating misinstalled pthreads executable" )
      file( COPY ${PTHREADS_DLL_WIN} DESTINATION ${OUTPUT_BIN_DIR} )
    endif()
  else()
    set( DARKNET_BIN_LINUX "${VIAME_BUILD_PREFIX}/src/darknet-build/darknet" )

    if( EXISTS ${DARKNET_BIN_LINUX} )
      message( WARNING "Relocating misinstalled darknet executable" )
      file( COPY ${DARKNET_BIN_LINUX} DESTINATION ${OUTPUT_BIN_DIR} )
    endif()
  endif()
endif()

# Check for files mis-installed into x64 folder
if( EXISTS "${VIAME_INSTALL_PREFIX}/x64/vc16" )
  file( COPY "${VIAME_INSTALL_PREFIX}/x64/vc16/bin" DESTINATION ${VIAME_INSTALL_PREFIX} )
  file( COPY "${VIAME_INSTALL_PREFIX}/x64/vc16/lib" DESTINATION ${VIAME_INSTALL_PREFIX} )

  file( REMOVE_RECURSE "${VIAME_INSTALL_PREFIX}/x64" )
endif()

# Remove mis-installed OpenCV files
if( EXISTS "${VIAME_INSTALL_PREFIX}/setup_vars_opencv4.cmd" )
  message( WARNING "Cleaning up OpenCV files" )
  file( REMOVE "${VIAME_INSTALL_PREFIX}/setup_vars_opencv4.cmd" )
endif()
if( EXISTS "${VIAME_INSTALL_PREFIX}/LICENSE" )
  file( REMOVE "${VIAME_INSTALL_PREFIX}/LICENSE" )
endif()
if( EXISTS "${VIAME_INSTALL_PREFIX}/OpenCVConfig.cmake" )
  file( COPY "${VIAME_INSTALL_PREFIX}/OpenCVConfig.cmake"
        DESTINATION "${VIAME_INSTALL_PREFIX}/cmake/OpenCVConfig.cmake" )
endif()
if( EXISTS "${VIAME_INSTALL_PREFIX}/OpenCVConfig-version.cmake" )
  file( COPY "${VIAME_INSTALL_PREFIX}/OpenCVConfig-version.cmake"
        DESTINATION "${VIAME_INSTALL_PREFIX}/cmake/OpenCVConfig-version.cmake" )
endif()

# Remove any root level files that don't belong here
file( GLOB ROOT_CMAKE_FILES "${VIAME_INSTALL_PREFIX}/*.cmake" )
file( GLOB ROOT_TXT_FILES "${VIAME_INSTALL_PREFIX}/*.txt" )

if( ROOT_TXT_FILES )
  file( REMOVE ${ROOT_TXT_FILES} )
endif()
if( ROOT_CMAKE_FILES )
  file( REMOVE ${ROOT_CMAKE_FILES} )
endif()

