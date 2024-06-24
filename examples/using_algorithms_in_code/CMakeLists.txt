###
# VIAME examples
##

###
# Install files to example directory
##
install( FILES       detector1.cxx
                     detector3.cxx
                     detector4.cxx
                     CMakeLists.txt
         DESTINATION examples/using_algorithms_in_code )


# Build select files
if( VIAME_ENABLE_KWIVER AND VIAME_ENABLE_OPENCV )

  find_package( OpenCV REQUIRED )
  include_directories( SYSTEM ${OpenCV_INCLUDE_DIRS} )

  find_package( Eigen3 REQUIRED )
  include_directories( ${EIGEN3_INCLUDE_DIR} )

  ###
  # Add first detector example
  add_executable( detector1       detector1.cxx )

  target_link_libraries( detector1
    PRIVATE    kwiver::vital
               kwiver::vital_algo
               kwiver::vital_config
               kwiver::vital_logger
               kwiver::vital_util
               kwiver::kwiver_algo_core
               kwiver::kwiver_algo_ocv
               ${OpenCV_LIBS}
    )


  ###
  # Add first detector example
  add_executable( detector3       detector3.cxx )

  target_link_libraries( detector3
    PRIVATE    kwiver::vital
               kwiver::vital_algo
               kwiver::vital_config
               kwiver::vital_logger
               kwiver::vital_util
               kwiver::vital_vpm
               kwiver::kwiver_algo_core
               kwiver::kwiver_algo_ocv
               ${OpenCV_LIBS}
    )

endif()
