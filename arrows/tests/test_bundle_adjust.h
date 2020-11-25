// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_scene.h>

#include <arrows/mvg/metrics.h>
#include <arrows/mvg/projected_track_set.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;
using namespace kwiver::arrows::mvg;

// ----------------------------------------------------------------------------
// Input to SBA is the ideal solution, make sure it doesn't diverge
TEST(bundle_adjust, from_solution)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  double init_rmse = reprojection_rmse(cameras->cameras(),
                                       landmarks->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_LE(init_rmse, 1e-12) << "Initial reprojection RMSE should be small";

  ba.optimize(cameras, landmarks, tracks);

  double end_rmse = reprojection_rmse(cameras->cameras(),
                                      landmarks->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-12) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks before input to SBA
TEST(bundle_adjust, noisy_landmarks)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  double init_rmse = reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks and cameras before input to SBA
TEST(bundle_adjust, noisy_landmarks_noisy_cameras)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1, 0.1);

  double init_rmse = reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Initialize all landmarks to the origin as input to SBA
TEST(bundle_adjust, zero_landmarks)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // initialize all landmarks to the origin
  landmark_id_t num_landmarks = static_cast<landmark_id_t>(landmarks->size());
  landmark_map_sptr landmarks0 = kwiver::testing::init_landmarks(num_landmarks);

  double init_rmse = reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks and cameras before input to SBA; select a subset of
// cameras to optimize
TEST(bundle_adjust, subset_cameras)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1, 0.1);

  camera_map::map_camera_t cam_map = cameras0->cameras();
  camera_map::map_camera_t cam_map2;
  for ( auto const& p : cam_map )
  {
    /// take every third camera
    if(p.first % 3 == 0)
    {
      cam_map2.insert(p);
    }
  }
  cameras0 = std::make_shared<simple_camera_map>(cam_map2);

  EXPECT_EQ(7, cameras0->size()) << "Reduced number of cameras";

  double init_rmse = reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks and cameras before input to SBA; select a subset of
// landmarks to optimize
TEST(bundle_adjust, subset_landmarks)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1, 0.1);

  // remove some landmarks
  landmark_map::map_landmark_t lm_map = landmarks0->landmarks();
  lm_map.erase(1);
  lm_map.erase(4);
  lm_map.erase(5);
  landmarks0 = std::make_shared<simple_landmark_map>(lm_map);

  EXPECT_EQ(5, landmarks0->size()) << "Reduced number of landmarks";

  double init_rmse = reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks and cameras before input to SBA; select a subset of
// tracks/track_states to constrain the problem
TEST(bundle_adjust, subset_tracks)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1, 0.1);

  // remove some tracks/track_states
  feature_track_set_sptr tracks0 = kwiver::testing::subset_tracks(tracks, 0.5);

  double init_rmse = reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks0->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks0);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks0->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after SBA";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks and cameras and tracks before input to SBA; select a
// subset of tracks/track_states to constrain the problem
TEST(bundle_adjust, noisy_tracks)
{
  bundle_adjust ba;
  kwiver::vital::config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("g_tolerance", "1e-12");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1, 0.1);

  // remove some tracks/track_states and add Gaussian noise
  const double track_stdev = 1.0;
  feature_track_set_sptr tracks0 = kwiver::testing::noisy_tracks(
                               kwiver::testing::subset_tracks(tracks, 0.5),
                               track_stdev);

  double init_rmse = reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks0->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks0);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks0->tracks());
  EXPECT_NEAR(0.0, end_rmse, track_stdev) << "RMSE after SBA";
}
