
# Old Example Checks
if( EXISTS ${VIAME_BUILD_INSTALL_PREFIX}/examples/detector_pipelines )
  message( FATAL_ERROR "Your examples directory is too old. As of VIAME 0.9.7, "
    "the examples directory has been refactored. Delete [build-dir]/install/examples, "
    "or alternatively consider a fresh build." )  
endif()
