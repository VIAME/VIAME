# VIAME image_ops Tests

project( test_viame_image_ops )

include( kwiver-test-setup )

kwiver_discover_gtests( viame image_ops
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# While VXL is still built, the overlap test compares against it directly
set( overlap_libraries viame_image_ops kwiver::vital GTest::gtest )

if( VIAME_ENABLE_VXL )
  find_package( VXL REQUIRED )
  include( ${VXL_CMAKE_DIR}/UseVXL.cmake )
  include_directories( SYSTEM ${VXL_CORE_INCLUDE_DIR} )
  include_directories( SYSTEM ${VXL_VCL_INCLUDE_DIR} )
  link_directories( ${VXL_LIBRARY_DIR} )
  list( APPEND overlap_libraries kwiver::kwiver_algo_vxl vnl )
endif()

kwiver_discover_gtests( viame homography_overlap
  LIBRARIES ${overlap_libraries} )

if( VIAME_ENABLE_VXL )
  target_compile_definitions( test-viame-homography_overlap
    PRIVATE VIAME_TEST_HAVE_VXL )
endif()
