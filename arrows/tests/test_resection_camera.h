// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_eigen.h>
#include <test_scene.h>

#include <arrows/mvg/projected_track_set.h>

#include <vital/math_constants.h>

using namespace kwiver::vital;
using namespace kwiver::arrows::mvg;

// ----------------------------------------------------------------------------
// Helper to test camera resection
template < typename... Args >
void
test_resection_camera(
  landmark_map_sptr const& landmarks, feature_track_set_sptr const& tracks,
  frame_id_t test_frame, camera_perspective_sptr const& expected_camera,
  double center_tolerance, double rotation_tolerance, Args... args )
{
  // Compute camera pose from the points and their projections
  resection_camera algo;
  vector< bool > inliers;
  auto const estimated_camera =
    algo.resection( test_frame, landmarks, tracks, args... );

  ASSERT_NE( nullptr, estimated_camera );

  // True and computed camera poses
  auto const& expected_rotation = expected_camera->rotation();
  auto const& estimated_rotation = estimated_camera->rotation();
  std::cout
    << "expected center:\n" << expected_camera->center().transpose() << '\n'
    << "estimated center:\n" << estimated_camera->center().transpose() << '\n'
    << "expected rotation:\n" << expected_rotation.matrix() << '\n'
    << "estimated rotation:\n" << estimated_rotation.matrix() << '\n';

  auto rotation_error = expected_rotation.inverse() * estimated_rotation;
  std::cout << "rotation error = "
            << rad_to_deg * rotation_error.angle() << " degrees" << std::endl;

  EXPECT_LT( rotation_error.angle(), rotation_tolerance );
  EXPECT_MATRIX_SIMILAR( expected_camera->center(),
                         estimated_camera->center(), center_tolerance );
}

// ----------------------------------------------------------------------------
template < typename Func >
void
test_resection_cameras(
  Func const& func, camera_map_sptr const& camera_map )
{
  for( auto const& camera_iter : camera_map->cameras() )
  {
    auto const test_frame = camera_iter.first;
    auto const& camera =
      dynamic_pointer_cast< camera_perspective >( camera_iter.second );

    SCOPED_TRACE( "At frame " + std::to_string( test_frame ) );
    ASSERT_NE( nullptr, camera );

    func( test_frame, camera );
  }
}

// ----------------------------------------------------------------------------
// Test camera resection with ideal points
TEST ( resection_camera, ideal_points )
{
  // Landmarks at random locations
  auto landmarks = kwiver::testing::init_landmarks( 128 );
  landmarks = kwiver::testing::noisy_landmarks( landmarks, 1.0 );

  // Camera sequence (elliptical path)
  auto camera_map = kwiver::testing::camera_seq();

  // Tracks from projections
  auto tracks = projected_tracks( landmarks, camera_map );

  // Do the test
  test_resection_cameras(
    [ & ]( frame_id_t test_frame, camera_perspective_sptr const& camera ){
      auto inliers = std::unordered_set< landmark_id_t >{};
      test_resection_camera( landmarks, tracks, test_frame, camera,
                             ideal_center_tolerance, ideal_rotation_tolerance,
                             camera->intrinsics(), &inliers );

      std::cout << "inlier count = " << inliers.size() << std::endl;
      EXPECT_EQ( landmarks->size(), inliers.size() )
        << "all points should be inliers";
    }, camera_map );
}

// ----------------------------------------------------------------------------
// Test camera resection with noisy points
TEST ( resection_camera, noisy_points_with_image_size )
{
  // Landmarks at random locations
  auto landmarks = kwiver::testing::init_landmarks( 128 );
  landmarks = kwiver::testing::noisy_landmarks( landmarks, 1.0 );

  // Camera sequence (elliptical path)
  auto camera_map = kwiver::testing::camera_seq();

  // Tracks from the projections with some random noise added
  auto tracks = projected_tracks( landmarks, camera_map );
  tracks = kwiver::testing::noisy_tracks( tracks, 0.5 );

  // Do the test
  test_resection_cameras(
    [ & ]( frame_id_t test_frame, camera_perspective_sptr const& camera ){
      test_resection_camera( landmarks, tracks, test_frame, camera,
                             noisy_center_tolerance, noisy_rotation_tolerance,
                             camera->image_width(), camera->image_height() );
    }, camera_map );
}

// ----------------------------------------------------------------------------
// Test camera resection with noisy points, missing tracks, and initial
// calibration guess
TEST ( resection_camera, noisy_points_with_initial_calibration )
{
  // Landmarks at random locations
  auto landmarks = kwiver::testing::init_landmarks( 128 );
  landmarks = kwiver::testing::noisy_landmarks( landmarks, 1.0 );

  // Camera sequence (elliptical path)
  auto camera_map = kwiver::testing::camera_seq();

  // Sparse tracks from the projections with some random noise added
  auto tracks = projected_tracks( landmarks, camera_map );
  tracks = kwiver::testing::noisy_tracks( tracks, 0.5 );
  tracks = kwiver::testing::subset_tracks( tracks, 0.2 );

  // Do the test
  test_resection_cameras(
    [ & ]( frame_id_t test_frame, camera_perspective_sptr const& camera ){
      test_resection_camera( landmarks, tracks, test_frame, camera,
                             noisy_center_tolerance, noisy_rotation_tolerance,
                             camera->intrinsics() );
    }, camera_map );
}
