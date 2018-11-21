/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include <test_scene.h>

#include <arrows/core/metrics.h>
#include <arrows/core/projected_track_set.h>
#include <arrows/core/triangulate_landmarks.h>
#include <vital/tests/rpc_reader.h>
#include <tests/test_gtest.h>

kwiver::vital::path_t g_data_dir;

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  GET_ARG(1, g_data_dir);
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class triangulate_landmarks_rpc : public ::testing::Test
{
  TEST_ARG(data_dir);

  triangulate_landmarks_rpc()
  {
    // Landmarks pulled from Google Maps
    std::vector< vector_3d > lm_pos;
    lm_pos.push_back( vector_3d( -117.237465, 32.881208, 110.0 ) );
    lm_pos.push_back( vector_3d( -117.235309, 32.879108, 110.0 ) );
    lm_pos.push_back( vector_3d( -117.239404, 32.877824, 110.0 ) );
    lm_pos.push_back( vector_3d( -117.236088, 32.877091, 110.0 ) );
    lm_pos.push_back( vector_3d( -117.240455, 32.876183, 110.0 ) );

    for ( size_t i = 0; i < lm_pos.size(); ++i )
    {
      auto landmark_ptr = std::make_shared< landmark_< double > >( lm_pos[i] );
      landmark_map.insert(
        std::pair< landmark_id_t, landmark_sptr >(i, landmark_ptr ) );
    }

    for ( size_t i = 0; i < 8; ++i )
    {
      path_t filepath = data_dir + "/rpc_data" + std::to_string(i) + ".dat";
      auto cam_ptr =
        std::make_shared< simple_camera_rpc >( read_rpc( filepath ) );
      camera_map.insert( std::pair< frame_id_t, camera_sptr >( i, cam_ptr ) );
    }
  }

  kwiver::vital::landmark_map::map_landmark_t landmark_map;
  kwiver::vital::camera_map::map_camera_t camera_map;
};

// ----------------------------------------------------------------------------
TEST_F(triangulate_landmarks_rpc, from_data)
{
  kwiver::arrows::core::triangulate_landmarks tri_lm;
  config_block_sptr cfg = tri_lm.get_configuration();

  landmark_map_sptr landmarks =
    std::make_shared< simple_landmark_map >( landmark_map );

  camera_map_sptr cameras = std::make_shared< simple_camera_map >( camera_map );

  auto tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  double init_rmse = kwiver::arrows::reprojection_rmse( cameras->cameras(),
                                                        landmarks->landmarks(),
                                                        tracks->tracks() );
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;

  EXPECT_LE(init_rmse, 1e-12) << "Initial reprojection RMSE should be small";

  tri_lm.triangulate(cameras, tracks, landmarks);

  double end_rmse = kwiver::arrows::reprojection_rmse( cameras->cameras(),
                                                       landmarks->landmarks(),
                                                       tracks->tracks() );
  EXPECT_NEAR(0.0, end_rmse, 0.05) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
TEST_F(triangulate_landmarks_rpc, noisy_tracks)
{
  kwiver::arrows::core::triangulate_landmarks tri_lm;
  config_block_sptr cfg = tri_lm.get_configuration();

  landmark_map_sptr landmarks =
    std::make_shared< simple_landmark_map >( landmark_map );

  camera_map_sptr cameras = std::make_shared< simple_camera_map >( camera_map );

  auto tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // remove some tracks/track_states and add Gaussian noise
  const double track_stdev = 1.0;
  feature_track_set_sptr tracks0 = kwiver::testing::noisy_tracks(
    kwiver::testing::subset_tracks(tracks, 0.5), track_stdev);

  double init_rmse = kwiver::arrows::reprojection_rmse( cameras->cameras(),
                                                        landmarks->landmarks(),
                                                        tracks0->tracks() );
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;

  EXPECT_LE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";;

  tri_lm.triangulate(cameras, tracks0, landmarks);

  double end_rmse = kwiver::arrows::reprojection_rmse( cameras->cameras(),
                                                       landmarks->landmarks(),
                                                       tracks0->tracks() );
  EXPECT_NEAR(0.0, end_rmse, 2.0*track_stdev) << "RMSE after triangulation";
}
