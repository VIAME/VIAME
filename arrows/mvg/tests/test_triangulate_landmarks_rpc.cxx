// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_scene.h>

#include <arrows/mvg/metrics.h>
#include <arrows/mvg/projected_track_set.h>
#include <arrows/mvg/algo/triangulate_landmarks.h>
#include <vital/tests/rpc_reader.h>
#include <tests/test_gtest.h>

kwiver::vital::path_t g_data_dir;

using namespace kwiver::vital;
using namespace kwiver::arrows::mvg;

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
  triangulate_landmarks tri_lm;
  config_block_sptr cfg = tri_lm.get_configuration();

  landmark_map_sptr landmarks =
    std::make_shared< simple_landmark_map >( landmark_map );

  camera_map_sptr cameras = std::make_shared< simple_camera_map >( camera_map );

  auto tracks = projected_tracks(landmarks, cameras);

  double init_rmse = reprojection_rmse( cameras->cameras(),
                                                        landmarks->landmarks(),
                                                        tracks->tracks() );
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;

  EXPECT_LE(init_rmse, 1e-12) << "Initial reprojection RMSE should be small";

  tri_lm.triangulate(cameras, tracks, landmarks);

  double end_rmse = reprojection_rmse( cameras->cameras(),
                                                       landmarks->landmarks(),
                                                       tracks->tracks() );
  EXPECT_NEAR(0.0, end_rmse, 0.05) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
TEST_F(triangulate_landmarks_rpc, noisy_tracks)
{
  triangulate_landmarks tri_lm;
  config_block_sptr cfg = tri_lm.get_configuration();

  landmark_map_sptr landmarks =
    std::make_shared< simple_landmark_map >( landmark_map );

  camera_map_sptr cameras = std::make_shared< simple_camera_map >( camera_map );

  auto tracks = projected_tracks(landmarks, cameras);

  // remove some tracks/track_states and add Gaussian noise
  const double track_stdev = 1.0;
  feature_track_set_sptr tracks0 = kwiver::testing::noisy_tracks(
    kwiver::testing::subset_tracks(tracks, 0.5), track_stdev);

  double init_rmse = reprojection_rmse( cameras->cameras(),
                                                        landmarks->landmarks(),
                                                        tracks0->tracks() );
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;

  EXPECT_LE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";;

  tri_lm.triangulate(cameras, tracks0, landmarks);

  double end_rmse = reprojection_rmse( cameras->cameras(),
                                                       landmarks->landmarks(),
                                                       tracks0->tracks() );
  EXPECT_NEAR(0.0, end_rmse, 2.0*track_stdev) << "RMSE after triangulation";
}
