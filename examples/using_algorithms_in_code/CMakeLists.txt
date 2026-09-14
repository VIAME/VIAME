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
#
# No OpenCV: these were written when the image container was the bridge's,
# and none of them ever used a `cv::` type -- the include and the link were
# left behind by P7.

###
# Add first detector example
add_executable( detector1       detector1.cxx )

viame_target_link_libraries( detector1
  PRIVATE    kwiver::vital
             kwiver::vital_algo
             kwiver::vital_config
             kwiver::vital_logger
             kwiver::vital_util
             viame_video_io
             viame_object_detectors
  )


###
# Add first detector example
add_executable( detector3       detector3.cxx )

viame_target_link_libraries( detector3
  PRIVATE    kwiver::vital
             kwiver::vital_algo
             kwiver::vital_config
             kwiver::vital_logger
             kwiver::vital_util
             kwiver::vital_vpm
             viame_video_io
             # It calls load_all_plugins, and since P8-T03 that means
             # linking what registers rather than finding it on a path
             viame::registry
  )

