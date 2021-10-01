message("Running smqtk post-install cleanup")

include( ${VIAME_CMAKE_DIR}/common_macros.cmake )

# SVM install tree quick hacks
if( NOT WIN32 )
  CreateSymlink( ${VIAME_INSTALL_PREFIX}/lib/libsvm.so.2
                 ${VIAME_INSTALL_PREFIX}/lib/libsvm.so )
endif()

message("Done")
