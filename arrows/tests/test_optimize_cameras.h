// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_eigen.h>
#include <test_scene.h>

#include <arrows/mvg/projected_track_set.h>

#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>

#include <iostream>

using namespace kwiver::vital;
using namespace kwiver::arrows::mvg;

// ----------------------------------------------------------------------------
TEST(optimize_cameras, uninitialized)
{
  camera_map_sptr cam_map;
  landmark_map_sptr lm_map;
  feature_track_set_sptr trk_set;

  optimize_cameras optimizer;
  config_block_sptr cfg = optimizer.get_configuration();
  cfg->set_value( "verbose", "true" );
  optimizer.set_configuration( cfg );

  EXPECT_THROW(
    optimizer.optimize( cam_map, trk_set, lm_map ),
    kwiver::vital::invalid_value )
    << "Running camera optimization with null input";

  EXPECT_EQ( nullptr, cam_map );
}

// ----------------------------------------------------------------------------
TEST(optimize_cameras, empty_input)
{
  camera_map_sptr cam_map = std::make_shared<simple_camera_map>();
  landmark_map_sptr lm_map = std::make_shared<simple_landmark_map>();
  feature_track_set_sptr trk_set = std::make_shared<feature_track_set>();

  optimize_cameras optimizer;
  config_block_sptr cfg = optimizer.get_configuration();
  cfg->set_value( "verbose", "true" );
  optimizer.set_configuration( cfg );

  camera_map_sptr orig_map = cam_map;

  std::cerr << "cam_map before: " << cam_map << std::endl;
  optimizer.optimize(cam_map, trk_set, lm_map);
  std::cerr << "cam_map after : " << cam_map << std::endl;
  std::cerr << "orig map      : " << orig_map << std::endl;

  // Make sure that a new camera map was created, but nothing was put in it.
  EXPECT_NE( orig_map, cam_map );
  EXPECT_EQ( 0, cam_map->size() );
  EXPECT_EQ( 0, orig_map->size() );
}

// ----------------------------------------------------------------------------
TEST(optimize_cameras, no_noise)
{
  // Create cameras, landmarks and tracks.
  // Optimize already optimimal elements to make sure they don't get changed
  // much.

  camera_map::map_camera_t original_cams = kwiver::testing::camera_seq()->cameras();

  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);
  camera_map_sptr working_cam_map =
    std::make_shared<simple_camera_map>( original_cams );
  feature_track_set_sptr tracks = projected_tracks( landmarks, working_cam_map );

  optimize_cameras optimizer;
  config_block_sptr cfg = optimizer.get_configuration();
  cfg->set_value( "verbose", "true" );
  optimizer.set_configuration( cfg );

  optimizer.optimize(working_cam_map, tracks, landmarks);

  double const ep = 1e-14;
  for( auto const& p : working_cam_map->cameras() )
  {
    SCOPED_TRACE( "At camera " + std::to_string( p.first ) );

    auto orig_cam_ptr =
      std::dynamic_pointer_cast<camera_perspective>( p.second );
    auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>( p.second );
    // Difference in camera center
    EXPECT_MATRIX_NEAR( orig_cam_ptr->center(), cam_ptr->center(), ep );

    // Difference in camera rotation
    EXPECT_MATRIX_NEAR(
      vector_4d{ orig_cam_ptr->rotation().quaternion().coeffs() },
      vector_4d{ cam_ptr->rotation().quaternion().coeffs() }, ep );

    // difference in camera intrinsics
    EXPECT_MATRIX_NEAR( orig_cam_ptr->intrinsics()->as_matrix(),
                        cam_ptr->intrinsics()->as_matrix(), ep );
  }
}

// ----------------------------------------------------------------------------
TEST(optimize_cameras, noisy_cameras)
{
  // Same as above, but create an analogous set of cameras with noise added.
  // Check that optimized cameras are close to the original cameras.

  camera_map::map_camera_t original_cams = kwiver::testing::camera_seq()->cameras();

  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);
  camera_map_sptr working_cam_map =
    std::make_shared<simple_camera_map>( original_cams );
  feature_track_set_sptr tracks = projected_tracks(landmarks, working_cam_map);

  working_cam_map = kwiver::testing::noisy_cameras(working_cam_map, 0.1, 0.1);

  optimize_cameras optimizer;
  config_block_sptr cfg = optimizer.get_configuration();
  cfg->set_value( "verbose", "true" );
  optimizer.set_configuration( cfg );

  optimizer.optimize(working_cam_map, tracks, landmarks);

  for( auto const& p : working_cam_map->cameras() )
  {
    SCOPED_TRACE( "At camera " + std::to_string( p.first ) );

    auto orig_cam_ptr =
      std::dynamic_pointer_cast<camera_perspective>( original_cams[p.first] );
    auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>( p.second );
    // Difference in camera center
    EXPECT_MATRIX_NEAR( orig_cam_ptr->center(),
                        cam_ptr->center(), noisy_center_tolerance );

    // Difference in camera rotation
    EXPECT_MATRIX_SIMILAR(
      vector_4d{ orig_cam_ptr->rotation().quaternion().coeffs() },
      vector_4d{ cam_ptr->rotation().quaternion().coeffs() },
      noisy_rotation_tolerance );

    // difference in camera intrinsics
    EXPECT_MATRIX_NEAR( orig_cam_ptr->intrinsics()->as_matrix(),
                        cam_ptr->intrinsics()->as_matrix(),
                        noisy_intrinsics_tolerance );
  }
}
