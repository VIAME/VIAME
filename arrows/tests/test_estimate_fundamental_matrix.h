// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_eigen.h>
#include <test_scene.h>

#include <arrows/mvg/projected_track_set.h>
#include <arrows/mvg/metrics.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <Eigen/LU>

using namespace kwiver::vital;
using namespace kwiver::arrows::mvg;

// ----------------------------------------------------------------------------
// Print epipolar distance of pairs of points given a fundamental matrix
static void
print_epipolar_distances(
  kwiver::vital::matrix_3x3d const& F,
  std::vector<kwiver::vital::vector_2d> const& right_pts,
  std::vector<kwiver::vital::vector_2d> const& left_pts)
{
  matrix_3x3d Ft = F.transpose();
  for(unsigned i=0; i<right_pts.size(); ++i)
  {
    const vector_2d& pr = right_pts[i];
    const vector_2d& pl = left_pts[i];
    vector_3d vr(pr.x(), pr.y(), 1.0);
    vector_3d vl(pl.x(), pl.y(), 1.0);
    vector_3d lr = F * vr;
    vector_3d ll = Ft * vl;
    double sr = 1.0 / sqrt(lr.x()*lr.x() + lr.y()*lr.y());
    double sl = 1.0 / sqrt(ll.x()*ll.x() + ll.y()*ll.y());
    // sum of point to epipolar line distance in both images
    double d = vr.dot(ll);
    std::cout <<" dist right = "<<d*sr<<"  dist left = "<<d*sl << std::endl;
  }
}

// ----------------------------------------------------------------------------
// Test fundamental matrix estimation with ideal points
TEST(estimate_fundamental_matrix, ideal_points)
{
  estimate_fundamental_matrix est_f;

  // create landmarks at the random locations
  landmark_map_sptr landmarks = kwiver::testing::init_landmarks(100);
  landmarks = kwiver::testing::noisy_landmarks(landmarks, 1.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  track_set_sptr tracks = projected_tracks(landmarks, cameras);

  const frame_id_t frame1 = 0;
  const frame_id_t frame2 = 10;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam1 = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  auto cam2 = std::dynamic_pointer_cast<camera_perspective>(cams[frame2]);
  camera_intrinsics_sptr cal1 = cam1->intrinsics();
  camera_intrinsics_sptr cal2 = cam2->intrinsics();

  // compute the true fundamental matrix from the cameras
  fundamental_matrix_sptr true_F = fundamental_matrix_from_cameras(*cam1, *cam2);

  // extract corresponding image points
  std::vector<vector_2d> pts1, pts2;
  for ( auto const& track : tracks->tracks() )
  {
    auto const fts1 =
      std::dynamic_pointer_cast<feature_track_state>( *track->find( frame1 ) );
    auto const fts2 =
      std::dynamic_pointer_cast<feature_track_state>( *track->find( frame2 ) );
    pts1.push_back( fts1->feature->loc() );
    pts2.push_back( fts2->feature->loc() );
  }

  // print the epipolar distances using this fundamental matrix
  print_epipolar_distances(true_F->matrix(), pts1, pts2);

  // compute the fundamental matrix from the corresponding points
  std::vector<bool> inliers;
  auto estimated_F = est_f.estimate( pts1, pts2, inliers, 1.5 );

  // compare true and computed fundamental matrices
  std::cout << "true F = " << *true_F << std::endl;
  std::cout << "Estimated F = "<< *estimated_F << std::endl;
  EXPECT_MATRIX_SIMILAR( true_F->matrix(), estimated_F->matrix(),
                         ideal_tolerance );

  std::cout << "num inliers " << inliers.size() << std::endl;
  EXPECT_EQ( pts1.size(), inliers.size() )
    << "All points should be inliers";
}

// ----------------------------------------------------------------------------
// Test fundamental matrix estimation with noisy points
TEST(estimate_fundamental_matrix, noisy_points)
{
  estimate_fundamental_matrix est_f;

  // create landmarks at the random locations
  landmark_map_sptr landmarks = kwiver::testing::init_landmarks(100);
  landmarks = kwiver::testing::noisy_landmarks(landmarks, 1.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  auto tracks = std::dynamic_pointer_cast<feature_track_set>(
    projected_tracks( landmarks, cameras ) );

  // add random noise to track image locations
  tracks = kwiver::testing::noisy_tracks(tracks, 0.5);

  const frame_id_t frame1 = 0;
  const frame_id_t frame2 = 10;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam1 = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  auto cam2 = std::dynamic_pointer_cast<camera_perspective>(cams[frame2]);
  camera_intrinsics_sptr cal1 = cam1->intrinsics();
  camera_intrinsics_sptr cal2 = cam2->intrinsics();

  // compute the true fundamental matrix from the cameras
  fundamental_matrix_sptr true_F = fundamental_matrix_from_cameras(*cam1, *cam2);

  // extract corresponding image points
  std::vector<vector_2d> pts1, pts2;
  for ( auto const& track : tracks->tracks() )
  {
    auto const fts1 =
      std::dynamic_pointer_cast<feature_track_state>( *track->find( frame1 ) );
    auto const fts2 =
      std::dynamic_pointer_cast<feature_track_state>( *track->find( frame2 ) );
    pts1.push_back( fts1->feature->loc() );
    pts2.push_back( fts2->feature->loc() );
  }

  print_epipolar_distances(true_F->matrix(), pts1, pts2);

  // compute the fundamental matrix from the corresponding points
  std::vector<bool> inliers;
  auto estimated_F = est_f.estimate( pts1, pts2, inliers, 1.5 );

  // compare true and computed fundamental matrices
  std::cout << "true F = " << *true_F << std::endl;
  std::cout << "Estimated F = "<< *estimated_F << std::endl;
  EXPECT_MATRIX_SIMILAR( true_F->matrix(), estimated_F->matrix(), 0.01 );

  std::cout << "num inliers " << inliers.size() << std::endl;
  EXPECT_GT( inliers.size(), pts1.size() / 2 )
    << "Not enough inliers";
}

// ----------------------------------------------------------------------------
// Test fundamental matrix estimation with outliers
TEST(estimate_fundamental_matrix, outlier_points)
{
  estimate_fundamental_matrix est_f;

  // create landmarks at the random locations
  landmark_map_sptr landmarks = kwiver::testing::init_landmarks(100);
  landmarks = kwiver::testing::noisy_landmarks(landmarks, 1.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  auto tracks = std::dynamic_pointer_cast<feature_track_set>(
    projected_tracks( landmarks, cameras ) );

  // add random noise to track image locations
  tracks = kwiver::testing::noisy_tracks(tracks, 0.5);

  const frame_id_t frame1 = 0;
  const frame_id_t frame2 = 10;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam1 = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  auto cam2 = std::dynamic_pointer_cast<camera_perspective>(cams[frame2]);
  camera_intrinsics_sptr cal1 = cam1->intrinsics();
  camera_intrinsics_sptr cal2 = cam2->intrinsics();

  // compute the true fundamental matrix from the cameras
  fundamental_matrix_sptr true_F = fundamental_matrix_from_cameras(*cam1, *cam2);

  // extract corresponding image points
  unsigned int i = 0;
  std::vector<vector_2d> pts1, pts2;
  for ( auto const& track : tracks->tracks() )
  {
    if ( ++i % 3 == 0 )
    {
      pts1.push_back( kwiver::testing::random_point2d( 1000.0 ) );
      pts2.push_back( kwiver::testing::random_point2d( 1000.0 ) );
    }
    else
    {
      auto const fts1 = std::dynamic_pointer_cast<feature_track_state>(
        *track->find( frame1 ) );
      auto const fts2 = std::dynamic_pointer_cast<feature_track_state>(
        *track->find( frame2 ) );
      pts1.push_back( fts1->feature->loc() );
      pts2.push_back( fts2->feature->loc() );
    }
  }

  print_epipolar_distances(true_F->matrix(), pts1, pts2);

  // compute the fundamental matrix from the corresponding points
  std::vector<bool> inliers;
  auto estimated_F = est_f.estimate( pts1, pts2, inliers, 1.5 );

  // compare true and computed fundamental matrices
  std::cout << "true F = " << *true_F << std::endl;
  std::cout << "Estimated F = "<< *estimated_F << std::endl;
  EXPECT_MATRIX_SIMILAR( true_F->matrix(), estimated_F->matrix(),
                         outlier_tolerance );

  std::cout << "num inliers " << inliers.size() << std::endl;
  EXPECT_GT( inliers.size(), pts1.size() / 3 )
    << "Not enough inliers";
}
