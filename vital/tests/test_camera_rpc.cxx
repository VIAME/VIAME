/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
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
 * \brief test core camera class
 */

#include <test_eigen.h>
#include <tests/test_gtest.h>

#include <vital/types/camera_rpc.h>
#include <vital/io/camera_io.h>
#include <vital/tests/rpc_reader.h>

#include <iostream>

static double epsilon = 1e-8;
kwiver::vital::path_t g_data_dir;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  GET_ARG(1, g_data_dir);
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class camera_rpc : public ::testing::Test
{
  TEST_ARG(data_dir);

public:
  void SetUp()
  {
    // Actual world values
    static const size_t num_points = 5;
    double act_x[num_points] =
      { -58.58940727826357, -58.589140738420539, -58.588819506933184, -58.58855693683482, -58.58839238727699 };
    double act_y[num_points] =
      { -34.49283455146763, -34.492818509990848, -34.492808611762605, -34.492802905977392, -34.49280925602671 };
    double act_z[num_points] =
      { 20.928231142319902, 21.9573811423199, 27.1871011423199, 19.2657311423199, 26.606641142319901 };

    // Actual projection values
    double act_u[num_points] =
      { 16581.12626986, 16519.24664854, 16449.76676766, 16377.35597454, 16347.72126206 };
    double act_v[num_points] =
      { 15443.08533878, 15451.02512727, 15458.40044985, 15461.20973047, 15462.29884238 };

    for (size_t i = 0; i<num_points; ++i)
    {
      test_points.push_back(
        kwiver::vital::vector_3d( act_x[i], act_y[i], act_z[i] ) );
      test_image_points.push_back(
        kwiver::vital::vector_2d( act_u[i], act_v[i] ) );
    }
  }

  std::vector<kwiver::vital::vector_3d> test_points;
  std::vector<kwiver::vital::vector_2d> test_image_points;
};

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, identity)
{
  kwiver::vital::simple_camera_rpc icam;

  kwiver::vital::vector_3d pt(1.0, 2.0, 10.0);

  auto img_pt = icam.project( pt );

  EXPECT_MATRIX_EQ( img_pt, kwiver::vital::vector_2d(1.0, 2.0) );
}

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, clone)
{
  kwiver::vital::path_t test_rpc_file = data_dir + "/rpc_data.dat";
  auto cam = read_rpc( test_rpc_file );
  auto cam_clone =
    std::dynamic_pointer_cast<kwiver::vital::camera_rpc>( cam.clone() );

  EXPECT_MATRIX_EQ( cam.world_scale(), cam_clone->world_scale() );
  EXPECT_MATRIX_EQ( cam.world_offset(), cam_clone->world_offset() );
  EXPECT_MATRIX_EQ( cam.image_scale(), cam_clone->image_scale() );
  EXPECT_MATRIX_EQ( cam.image_offset(), cam_clone->image_offset() );
  EXPECT_MATRIX_EQ( cam.rpc_coeffs(), cam_clone->rpc_coeffs() );
  EXPECT_EQ( cam.image_width(), cam_clone->image_width() );
  EXPECT_EQ( cam.image_height(), cam_clone->image_height() );
}

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, projection)
{
  kwiver::vital::path_t test_rpc_file = data_dir + "/rpc_data.dat";
  auto cam = read_rpc( test_rpc_file );

  for (size_t i = 0; i < test_points.size(); ++i)
  {
    auto img_pt = cam.project( test_points[i] );

    EXPECT_MATRIX_NEAR( img_pt, test_image_points[i], epsilon );
  }
}

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, back_projection)
{
  kwiver::vital::path_t test_rpc_file = data_dir + "/rpc_data.dat";
  auto cam = read_rpc( test_rpc_file );

  for (size_t i = 0; i < test_points.size(); ++i)
  {
    auto img_pt = cam.project( test_points[i] );
    auto new_pt = cam.back_project( img_pt, test_points[i][2] );

    EXPECT_MATRIX_NEAR( new_pt, test_points[i], epsilon );
  }
}

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, read_missing_image_dimension)
{
  kwiver::vital::path_t test_rpc_file = data_dir + "/rpc_data_missing_image_dimension.dat";
  auto cam = read_rpc( test_rpc_file );
  EXPECT_EQ( cam.image_width(), 0 );
  EXPECT_EQ( cam.image_height(), 0 );
}
