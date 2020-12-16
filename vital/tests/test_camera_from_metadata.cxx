/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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
 * \brief test creation of rpc camera class from metadata
 */

#include <test_eigen.h>

#include <vital/io/camera_from_metadata.h>
#include <vital/types/camera_rpc.h>
#include <vital/types/metadata_traits.h>

#include <iostream>

using namespace kwiver::vital;

namespace
{
  // sample RPC metadata read by GDAL from the following 2016 MVS Benchmark image
  // 01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF
  constexpr double HEIGHT_OFFSET = 31.0;
  constexpr double HEIGHT_SCALE = 501.0;
  constexpr double LAT_OFFSET = -34.4732;
  constexpr double LAT_SCALE = 0.0708;
  constexpr double LONG_OFFSET = -58.6096;
  constexpr double LONG_SCALE = 0.0928;
  constexpr double ROW_OFFSET = 21477.0;
  constexpr double ROW_SCALE = 21478.0;
  constexpr double COL_OFFSET = 21249.0;
  constexpr double COL_SCALE = 21250.0;

  std::string const row_num_coeff =
    "0.0002703625 0.04284488 1.046869 0.004713542 "
    "-0.0001706129 -1.525177e-07 1.255623e-05 -0.0005820134 "
    "-0.000710512 -2.510676e-07 3.179984e-06 3.120413e-06 "
    "3.19923e-05 4.194369e-06 7.475295e-05 0.0003630791 "
    "0.0001021649 4.493725e-07 3.156566e-06 4.596505e-07 ";

  std::string const row_den_coeff =
    "1 0.0001912806 0.0005166397 -1.45044e-05 "
    "-3.860133e-05 2.634582e-06 -4.551145e-06 6.859296e-05 "
    "-0.0002410782 9.753265e-05 -1.456261e-07 5.310624e-08 "
    "-1.913253e-05 3.18203e-08 3.870586e-07 -0.000206842 "
    "9.128349e-08 0 -2.506197e-06 0 ";

  std::string const col_num_coeff =
    "0.006585953 -1.032582 0.001740937 0.03034485 "
    "0.0008819178 -0.000167943 0.0001519299 -0.00626254 "
    "-0.00107337 9.099077e-06 2.608985e-06 -2.947004e-05 "
    "2.231277e-05 4.587831e-06 4.16379e-06 0.0003464555 "
    "3.598323e-08 -2.859541e-06 5.159311e-06 -1.349187e-07 ";

  std::string const col_den_coeff =
    "1 0.0003374458 0.0008965622 -0.0003730697 "
    "-2.666499e-05 -2.711356e-06 5.454434e-07 4.485658e-07 "
    "2.534922e-05 -4.546709e-06 0 -1.056044e-07 "
    "-5.626866e-07 2.243313e-08 -2.108053e-07 9.199534e-07 "
    "0 -3.887594e-08 -1.437016e-08 0 ";
}

metadata_sptr generate_metadata()
{
  metadata_sptr md = std::make_shared<metadata>();

  // sample RPC metadata read by GDAL from the following 2016 MVS Benchmark image
  // 01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF
  md->add< VITAL_META_RPC_HEIGHT_OFFSET >( HEIGHT_OFFSET );
  md->add< VITAL_META_RPC_HEIGHT_SCALE >( HEIGHT_SCALE );
  md->add< VITAL_META_RPC_LAT_OFFSET >( LAT_OFFSET );
  md->add< VITAL_META_RPC_LAT_SCALE >( LAT_SCALE );
  md->add< VITAL_META_RPC_LONG_OFFSET >( LONG_OFFSET );
  md->add< VITAL_META_RPC_LONG_SCALE >( LONG_SCALE );

  md->add< VITAL_META_RPC_ROW_OFFSET >( ROW_OFFSET );
  md->add< VITAL_META_RPC_ROW_SCALE >( ROW_SCALE );
  md->add< VITAL_META_RPC_COL_OFFSET >( COL_OFFSET );
  md->add< VITAL_META_RPC_COL_SCALE >( COL_SCALE );

  md->add< VITAL_META_RPC_ROW_NUM_COEFF >( row_num_coeff );
  md->add< VITAL_META_RPC_ROW_DEN_COEFF >( row_den_coeff );
  md->add< VITAL_META_RPC_COL_NUM_COEFF >( col_num_coeff );
  md->add< VITAL_META_RPC_COL_DEN_COEFF >( col_den_coeff );

  return md;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

TEST(camera_from_metadata, invalid_metadata)
{
  metadata_sptr rpc_metadata = std::make_shared<metadata>();

  EXPECT_THROW(
    camera_rpc_sptr cam = std::dynamic_pointer_cast<camera_rpc>(
      camera_from_metadata( rpc_metadata ) ),
    kwiver::vital::metadata_exception );
}

// ----------------------------------------------------------------------------
TEST(camera_from_metadata, valid_metadata)
{
  auto rpc_metadata = generate_metadata();

  vector_3d world_scale( LONG_SCALE, LAT_SCALE, HEIGHT_SCALE );
  vector_3d world_offset( LONG_OFFSET, LAT_OFFSET, HEIGHT_OFFSET );
  vector_2d image_scale( ROW_SCALE, COL_SCALE );
  vector_2d image_offset( ROW_OFFSET, COL_OFFSET );

  rpc_matrix rpc_coeff;
  rpc_coeff.row(0) = string_to_vector( row_num_coeff );
  rpc_coeff.row(1) = string_to_vector( row_den_coeff );
  rpc_coeff.row(2) = string_to_vector( col_num_coeff );
  rpc_coeff.row(3) = string_to_vector( col_den_coeff );

  camera_rpc_sptr cam = std::dynamic_pointer_cast<camera_rpc>(
    camera_from_metadata( rpc_metadata ) );

  EXPECT_MATRIX_EQ( cam->world_scale(), world_scale );
  EXPECT_MATRIX_EQ( cam->world_offset(), world_offset );
  EXPECT_MATRIX_EQ( cam->image_scale(), image_scale );
  EXPECT_MATRIX_EQ( cam->image_offset(), image_offset );
  EXPECT_MATRIX_EQ( cam->rpc_coeffs(), rpc_coeff );
}

TEST(camera_from_metadata, projection)
{
  auto rpc_metadata = generate_metadata();
  auto cam = camera_from_metadata( rpc_metadata );

  std::vector<vector_3d> wld_pts;
  std::vector<vector_2d> img_pts;

  wld_pts.push_back(
    vector_3d( -58.589407278263572, -34.492834551467631, 20.928231142319902 ) );
  wld_pts.push_back(
    vector_3d( -58.589140738420539, -34.492818509990848, 21.9573811423199 ) );
  wld_pts.push_back(
    vector_3d( -58.588819506933184, -34.492808611762605, 27.1871011423199 ) );
  wld_pts.push_back(
    vector_3d( -58.58855693683482, -34.492802905977392, 19.2657311423199 ) );
  wld_pts.push_back(
    vector_3d( -58.58839238727699, -34.49280925602671, 26.606641142319901 ) );

  img_pts.push_back( vector_2d( 15443.08533878, 16581.12626986 ) );
  img_pts.push_back( vector_2d( 15451.02512727, 16519.24664854 ) );
  img_pts.push_back( vector_2d( 15458.40044985, 16449.76676766 ) );
  img_pts.push_back( vector_2d( 15461.20973047, 16377.35597454 ) );
  img_pts.push_back( vector_2d( 15462.29884238, 16347.72126206 ) );

  for (unsigned int i=0; i<wld_pts.size(); ++i)
  {
    auto img_pt = cam->project( wld_pts[i] );
    EXPECT_MATRIX_EQ( img_pt, img_pts[i] );
  }
}

