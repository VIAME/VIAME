// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core camera_io tests
 */

#include <tests/test_eigen.h>
#include <tests/test_gtest.h>

#include <vital/io/camera_io.h>
#include <vital/exceptions.h>

#include <iostream>
#include <sstream>

kwiver::vital::path_t g_data_dir;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class camera_io : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(camera_io, KRTD_format_read)
{
  kwiver::vital::path_t test_read_file = data_dir + "/test_camera_io-valid_format.krtd";
  kwiver::vital::camera_perspective_sptr read_camera =
    kwiver::vital::read_krtd_file( test_read_file );

  Eigen::Matrix<double,3,3> expected_intrinsics;
  expected_intrinsics << 1, 2, 3,
                         0, 5, 6,
                         0, 0, 1;
  Eigen::Matrix<double,3,3> K( read_camera->intrinsics()->as_matrix() );
  EXPECT_MATRIX_EQ( expected_intrinsics, K );

  Eigen::Matrix<double,3,3> expected_rotation;
  expected_rotation << 1, 0, 0,
                       0, 1, 0,
                       0, 0, 1;
  Eigen::Matrix<double,3,3> R( read_camera->rotation().matrix() );
  EXPECT_MATRIX_EQ( expected_rotation, R );

  Eigen::Matrix<double,3,1> expected_translation;
  expected_translation << 1, 2, 3;
  Eigen::Matrix<double,3,1> T( read_camera->translation() );
  EXPECT_MATRIX_EQ( expected_translation, T );
  std::vector<double> expected_distortion = {1, 2, 3, 4, 5};
  std::vector<double> D = read_camera->intrinsics()->dist_coeffs() ;
  EXPECT_EQ( expected_distortion, D );
}

// ----------------------------------------------------------------------------
TEST_F(camera_io, invalid_file_path)
{
  EXPECT_THROW(
    kwiver::vital::read_krtd_file( data_dir + "/not_a_file.blob" ),
    kwiver::vital::file_not_found_exception )
    << "Tried loading an invalid file path";
}

// ----------------------------------------------------------------------------
TEST_F(camera_io, invalid_file_content)
{
  kwiver::vital::path_t invalid_content_file = data_dir + "/test_camera_io-invalid_file.krtd";
  EXPECT_THROW(
    kwiver::vital::camera_perspective_sptr cam =
      kwiver::vital::read_krtd_file( invalid_content_file ),
    kwiver::vital::invalid_data )
    << "Tried loading a file with invalid data";
}

// ----------------------------------------------------------------------------
TEST_F(camera_io, output_format_test)
{
  kwiver::vital::simple_camera_perspective cam;
  std::cerr << "Default constructed camera\n" << cam << std::endl;
  std::cerr << "cam.get_center()     : " << kwiver::vital::vector_3d(cam.get_center()).transpose() << std::endl;
  std::cerr << "cam.get_rotation()   : " << cam.get_rotation() << std::endl;
  std::cerr << "cam.get_translation(): " << cam.translation() << std::endl;

  // We're expecting -0's as this is what Eigen likes to output when a zero
  // vector is negated.
  std::stringstream ss;
  ss << cam;
  EXPECT_EQ(
    "1 0 0\n"
    "0 1 0\n"
    "0 0 1\n"
    "\n"
    "1 0 0\n"
    "0 1 0\n"
    "0 0 1\n"
    "\n"
    "-0 -0 -0\n"
    "\n"
    "0\n",
    ss.str() )
    << "Camera output string test";
}
