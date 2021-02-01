// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL bundle adjustment functionality
 */

#include <arrows/vxl/bundle_adjust.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

using kwiver::arrows::vxl::bundle_adjust;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(bundle_adjust, create)
{
  plugin_manager::instance().load_all_plugins();

  EXPECT_NE( nullptr, algo::bundle_adjust::create("vxl") );
}

// ----------------------------------------------------------------------------
#include <arrows/tests/test_bundle_adjust.h>

// ----------------------------------------------------------------------------
// Initialize all landmarks to the origin and all cameras to same location as
// input to SBA
TEST(bundle_adjust, zero_landmarks_same_cameras)
{
  using namespace kwiver::arrows;
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

  // initialize all cameras to at (0,0,1) looking at the origin
  frame_id_t num_cameras = static_cast<frame_id_t>(cameras->size());
  camera_map_sptr cameras0 = kwiver::testing::init_cameras(num_cameras);

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
