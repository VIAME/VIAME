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

#include <vital/types/camera_rpc.h>
#include <vital/io/camera_io.h>

#include <iostream>

static double epsilon = 1e-8;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class camera_rpc : public ::testing::Test
{
public:
  void SetUp()
  {
    rpc_coeffs = kwiver::vital::rpc_matrix::Zero();
    // sample RPC metadata read by GDAL from the following 2016 MVS Benchmark image
    // 01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF
    rpc_coeffs.row(0) << 0.006585953, -1.032582, 0.001740937, 0.03034485,
                    0.0008819178, -0.000167943, 0.0001519299, -0.00626254,
                    -0.00107337, 9.099077e-06, 2.608985e-06, -2.947004e-05,
                    2.231277e-05, 4.587831e-06, 4.16379e-06, 0.0003464555,
                    3.598323e-08, -2.859541e-06, 5.159311e-06, -1.349187e-07;

    rpc_coeffs.row(1) << 1, 0.0003374458, 0.0008965622, -0.0003730697,
                    -2.666499e-05, -2.711356e-06, 5.454434e-07, 4.485658e-07,
                    2.534922e-05, -4.546709e-06, 0, -1.056044e-07,
                    -5.626866e-07, 2.243313e-08, -2.108053e-07, 9.199534e-07,
                    0, -3.887594e-08, -1.437016e-08, 0;

    rpc_coeffs.row(2) << 0.0002703625, 0.04284488, 1.046869, 0.004713542,
                    -0.0001706129, -1.525177e-07, 1.255623e-05, -0.0005820134,
                    -0.000710512, -2.510676e-07, 3.179984e-06, 3.120413e-06,
                    3.19923e-05, 4.194369e-06, 7.475295e-05, 0.0003630791,
                    0.0001021649, 4.493725e-07, 3.156566e-06, 4.596505e-07;

    rpc_coeffs.row(3) << 1, 0.0001912806, 0.0005166397, -1.45044e-05,
                  -3.860133e-05, 2.634582e-06, -4.551145e-06, 6.859296e-05,
                  -0.0002410782, 9.753265e-05, -1.456261e-07, 5.310624e-08,
                  -1.913253e-05, 3.18203e-08, 3.870586e-07, -0.000206842,
                  9.128349e-08, 0, -2.506197e-06, 0;

    world_scale << 0.0928, 0.0708, 501;
    world_offset << -58.6096, -34.4732, 31;
    image_scale << 21250, 21478;
    image_offset << 21249, 21477;

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

  kwiver::vital::vector_3d world_scale;
  kwiver::vital::vector_3d world_offset;
  kwiver::vital::vector_2d image_scale;
  kwiver::vital::vector_2d image_offset;

  kwiver::vital::rpc_matrix rpc_coeffs;

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
  kwiver::vital::simple_camera_rpc cam( world_scale, world_offset,
                                        image_scale, image_offset, rpc_coeffs );
  auto cam_clone =
    std::dynamic_pointer_cast<kwiver::vital::camera_rpc>( cam.clone() );

  EXPECT_MATRIX_EQ( cam.world_scale(), cam_clone->world_scale() );
  EXPECT_MATRIX_EQ( cam.world_offset(), cam_clone->world_offset() );
  EXPECT_MATRIX_EQ( cam.image_scale(), cam_clone->image_scale() );
  EXPECT_MATRIX_EQ( cam.image_offset(), cam_clone->image_offset() );
  EXPECT_MATRIX_EQ( cam.rpc_coeffs(), cam_clone->rpc_coeffs() );
}

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, projection)
{
  kwiver::vital::simple_camera_rpc cam( world_scale, world_offset,
                                        image_scale, image_offset, rpc_coeffs );

  for (size_t i = 0; i < test_points.size(); ++i)
  {
    auto img_pt = cam.project( test_points[i] );

    EXPECT_MATRIX_NEAR( img_pt, test_image_points[i], epsilon );
  }
}

// ----------------------------------------------------------------------------
TEST_F(camera_rpc, back_projection)
{
  kwiver::vital::simple_camera_rpc cam( world_scale, world_offset,
                                        image_scale, image_offset, rpc_coeffs );

  for (size_t i = 0; i < test_points.size(); ++i)
  {
    auto img_pt = cam.project( test_points[i] );
    auto new_pt = cam.back_project( img_pt, test_points[i][2] );

    EXPECT_MATRIX_NEAR( new_pt, test_points[i], epsilon );
  }
}
