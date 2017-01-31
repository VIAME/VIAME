/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief core camera_io tests
 */

#include <test_common.h>

#include <iostream>
#include <sstream>

#include <vital/io/camera_io.h>
#include <vital/exceptions.h>


#define TEST_ARGS ( kwiver::vital::path_t const &data_dir )
DECLARE_TEST_MAP();


int main(int argc, char** argv)
{
  CHECK_ARGS(2);
  testname_t const testname = argv[1];
  kwiver::vital::path_t data_dir( argv[2] );
  RUN_TEST( testname, data_dir );
}


IMPLEMENT_TEST(KRTD_format_read)
{
  kwiver::vital::path_t test_read_file = data_dir + "/test_camera_io-valid_format.krtd";
  kwiver::vital::camera_sptr read_camera = kwiver::vital::read_krtd_file( test_read_file );

  Eigen::Matrix<double,3,3> expected_intrinsics;
  expected_intrinsics << 1, 2, 3,
                         0, 5, 6,
                         0, 0, 1;
  Eigen::Matrix<double,3,3> K( read_camera->intrinsics()->as_matrix() );
  std::cerr << "Read in K: " << K << std::endl;
  TEST_EQUAL( "read camera intrinsics",
              K.isApprox( expected_intrinsics ), true );

  Eigen::Matrix<double,3,3> expected_rotation;
  expected_rotation << 1, 0, 0,
                       0, 1, 0,
                       0, 0, 1;
  Eigen::Matrix<double,3,3> R( read_camera->rotation().matrix() );
  std::cerr << "Read in R: " << R << std::endl;
  TEST_EQUAL( "read camera rotation",
              R.isApprox( expected_rotation ), true );

  Eigen::Matrix<double,3,1> expected_translation;
  expected_translation << 1, 2, 3;
  Eigen::Matrix<double,3,1> T( read_camera->translation() );
  std::cerr << "Read in T: " << T << std::endl;
  TEST_EQUAL( "read camera translation",
              T.isApprox( expected_translation ), true );
  std::vector<double> expected_distortion = {1, 2, 3, 4, 5};
  std::vector<double> D = read_camera->intrinsics()->dist_coeffs() ;
  std::cerr << "Read in D: ";
  for (size_t i = 0; i < D.size(); ++i)
  {
    std::cerr << D[i] << std::endl;
  }
  TEST_EQUAL( "read camera distortion",
              D == expected_distortion, true );
}


IMPLEMENT_TEST(invalid_file_path)
{
  EXPECT_EXCEPTION(
      kwiver::vital::file_not_found_exception,
      kwiver::vital::read_krtd_file( data_dir + "/not_a_file.blob" ),
      "tried loading an invalid file path"
      );
}


IMPLEMENT_TEST(invalid_file_content)
{
  kwiver::vital::path_t invalid_content_file = data_dir + "/test_camera_io-invalid_file.krtd";
  EXPECT_EXCEPTION(
      kwiver::vital::invalid_data,
      kwiver::vital::camera_sptr cam = kwiver::vital::read_krtd_file( invalid_content_file ),
      "tried loading a file with invalid data"
      );
}


IMPLEMENT_TEST(output_format_test)
{
  kwiver::vital::simple_camera cam;
  std::cerr << "Default constructed camera\n" << cam << std::endl;
  std::cerr << "cam.get_center()     : " << kwiver::vital::vector_3d(cam.get_center()).transpose() << std::endl;
  std::cerr << "cam.get_rotation()   : " << cam.get_rotation() << std::endl;
  std::cerr << "cam.get_translation(): " << cam.translation() << std::endl;

  // We're expecting -0's as this is what Eigen likes to output when a zero
  // vector is negated.
  std::stringstream ss;
  ss << cam;
  TEST_EQUAL(
      "camera output string test",
      ss.str(),
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
      "0\n"
      );
}
