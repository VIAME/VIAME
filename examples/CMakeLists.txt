###
# VIAME examples


find_package( OpenCV 2.4.6 REQUIRED )
include_directories( SYSTEM ${OpenCV_INCLUDE_DIRS} )

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

###
# Add first detector example
add_executable( detector1       detector1.cxx )

target_link_libraries( detector1
  PRIVATE    vital
             kwiver_algo
             kwiver_algo_ocv
            ${OpenCV_LIBS}
  )


###
# Add first detector example
add_executable( detector3       detector3.cxx )

target_link_libraries( detector3
  PRIVATE    vital
             vital_apm
             kwiver_algo
             kwiver_algo_ocv
            ${OpenCV_LIBS}
  )
