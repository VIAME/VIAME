message( "Running fletch install" )

include( ${VIAME_CMAKE_DIR}/common_macros.cmake )

if( WIN32 )

  if( MSVC AND MSVC_VERSION EQUAL 1900 )
    RenameSubstr( ${VIAME_BUILD_INSTALL_PREFIX}/lib/libboost* vc120 vc140 )
  endif()
 
  if( VIAME_ENABLE_OPENCV )
    CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x86/*/bin/*.dll ${VIAME_BUILD_INSTALL_PREFIX}/bin )
    CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x86/*/lib/*.lib ${VIAME_BUILD_INSTALL_PREFIX}/lib )

    CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x64/*/bin/*.dll ${VIAME_BUILD_INSTALL_PREFIX}/bin )
    CopyFiles( ${VIAME_BUILD_INSTALL_PREFIX}/x64/*/lib/*.lib ${VIAME_BUILD_INSTALL_PREFIX}/lib )
  endif()

  if( VIAME_ENABLE_CAFFE )
    MoveFiles( ${VIAME_BUILD_INSTALL_PREFIX}/lib/caffe*.dll ${VIAME_BUILD_INSTALL_PREFIX}/bin )
  endif()
endif()

# Current caffe quick hacks
if( CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT WIN32 AND VIAME_ENABLE_CAFFE )
  CreateSymlink( ${VIAME_BUILD_INSTALL_PREFIX}/lib/libcaffe-d.so
                 ${VIAME_BUILD_INSTALL_PREFIX}/lib/libcaffe.so )
endif()

if( NOT WIN32 AND VIAME_ENABLE_CAFFE )
  CreateSymlink( ${VIAME_BUILD_INSTALL_PREFIX}/lib/libleveldb.so
                 ${VIAME_BUILD_INSTALL_PREFIX}/lib/libleveldb.so.1 )
endif()

# Move any misinstalled python files
if( PYTHON_VERSION )
  set( OUTPUT_PYTHON_DIR "${VIAME_BUILD_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/" )

  if( EXISTS ${VIAME_BUILD_INSTALL_PREFIX}/lib/site-packages )
    set( DIR_TO_MOVE "${VIAME_BUILD_INSTALL_PREFIX}/lib/site-packages" )
    file( GLOB FILES_TO_MOVE "${DIR_TO_MOVE}/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE  ${DIR_TO_MOVE} )
  endif()

  if( EXISTS ${VIAME_BUILD_INSTALL_PREFIX}/lib/python/site-packages )
    set( DIR_TO_MOVE "${VIAME_BUILD_INSTALL_PREFIX}/lib/python/site-packages" )
    file( GLOB FILES_TO_MOVE "${DIR_TO_MOVE}/*" )
    file( COPY ${FILES_TO_MOVE} DESTINATION ${OUTPUT_PYTHON_DIR} )
    file( REMOVE_RECURSE  ${DIR_TO_MOVE} )
  endif()

  if( NOT WIN32 AND VIAME_ENABLE_OPENCV )
    set( PATCH_DIR ${VIAME_CMAKE_DIR}/../packages/patches/fletch )
    file( COPY ${PATCH_DIR}/opencv_python-3.4.0.14.dist-info DESTINATION ${OUTPUT_PYTHON_DIR} )
  endif()
endif()

message( "Done" )
