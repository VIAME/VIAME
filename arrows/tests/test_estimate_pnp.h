// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <algorithm>

#include <test_eigen.h>
#include <test_scene.h>

#include <vital/math_constants.h>

#include <arrows/mvg/projected_track_set.h>
#include <arrows/mvg/metrics.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <Eigen/LU>

using namespace kwiver::vital;
using namespace kwiver::arrows::mvg;

// ----------------------------------------------------------------------------
// Test pnp pose estimation with ideal points
TEST(estimate_pnp, ideal_points)
{
  estimate_pnp est_pnp;

  // create landmarks at the random locations
  landmark_map_sptr landmarks = kwiver::testing::init_landmarks(100);
  landmarks = kwiver::testing::noisy_landmarks(landmarks, 1.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  track_set_sptr tracks = projected_tracks(landmarks, cameras);

  const frame_id_t frame1 = 0;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  camera_intrinsics_sptr cal = cam->intrinsics();

  // extract corresponding image points
  std::vector<vector_2d> pts_projs;
  std::vector<vector_3d> pts_3d;
  for (auto const& track : tracks->tracks())
  {
    auto lm_id = track->id();
    auto lm_it = landmarks->landmarks().find(lm_id);
    pts_3d.push_back(lm_it->second->loc());

    auto const fts =
      std::dynamic_pointer_cast<feature_track_state>(*track->find(frame1));
    pts_projs.push_back(fts->feature->loc());
  }

  // compute the camera pose from the points and their projections
  std::vector<bool> inliers;
  auto est_cam = est_pnp.estimate(pts_projs, pts_3d, cam->intrinsics(), inliers);

  // compare true and computed camera poses
  std::cout << "true R = " << cam->rotation().matrix() << std::endl;
  std::cout << "Estimated R = " << est_cam->rotation().matrix() << std::endl;
  std::cout << "true C = " << cam->center() << std::endl;
  std::cout << "Estimated C = " << est_cam->center() << std::endl;

  auto R_err = cam->rotation().inverse()*est_cam->rotation();
  std::cout << "Rotation error = " << R_err.angle()*180.0 / pi << " degrees" << std::endl;

  EXPECT_LT(R_err.angle(), ideal_rotation_tolerance);
  EXPECT_MATRIX_SIMILAR(cam->center(), est_cam->center(), ideal_center_tolerance);

  unsigned int num_inliers = std::count(inliers.begin(), inliers.end(), true);
  std::cout << "num inliers " << num_inliers << std::endl;
  EXPECT_EQ(pts_projs.size(), num_inliers)
    << "All points should be inliers";
}

// ----------------------------------------------------------------------------
// Test pnp pose estimation with noisy points
TEST(estimate_pnp, noisy_points)
{
  estimate_pnp est_pnp;

  // create landmarks at the random locations
  landmark_map_sptr landmarks = kwiver::testing::init_landmarks(100);
  landmarks = kwiver::testing::noisy_landmarks(landmarks, 1.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  auto tracks = std::dynamic_pointer_cast<feature_track_set>(
    projected_tracks(landmarks, cameras));

  // add random noise to track image locations
  tracks = kwiver::testing::noisy_tracks(tracks, 0.5);

  const frame_id_t frame1 = 1;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  camera_intrinsics_sptr cal = cam->intrinsics();

  // extract corresponding image points
  std::vector<vector_2d> pts_projs;
  std::vector<vector_3d> pts_3d;
  for (auto const& track : tracks->tracks())
  {
    auto lm_id = track->id();
    auto lm_it = landmarks->landmarks().find(lm_id);
    pts_3d.push_back(lm_it->second->loc());

    auto const fts =
      std::dynamic_pointer_cast<feature_track_state>(*track->find(frame1));
    pts_projs.push_back(fts->feature->loc());
  }

  // compute the camera pose from the points and their projections
  std::vector<bool> inliers;
  auto est_cam = est_pnp.estimate(pts_projs, pts_3d, cam->intrinsics(), inliers);

  // compare true and computed camera poses
  std::cout << "true R = " << cam->rotation().matrix() << std::endl;
  std::cout << "Estimated R = " << est_cam->rotation().matrix() << std::endl;
  std::cout << "true C = " << cam->center() << std::endl;
  std::cout << "Estimated C = " << est_cam->center() << std::endl;

  auto R_err = cam->rotation().inverse()*est_cam->rotation();
  std::cout << "Rotation error = " << R_err.angle()*180.0 / pi << " degrees" << std::endl;

  EXPECT_LT(R_err.angle(), noisy_rotation_tolerance);
  EXPECT_MATRIX_SIMILAR(cam->center(), est_cam->center(), noisy_center_tolerance);

  unsigned int num_inliers = std::count(inliers.begin(), inliers.end(), true);
  std::cout << "num inliers " << num_inliers << std::endl;
  EXPECT_GT(num_inliers, pts_projs.size() / 2)
    << "Not enough inliers";
}

// ----------------------------------------------------------------------------
// Test pnp pose estimation with outliers
TEST(estimate_pnp, outlier_points)
{
  estimate_pnp est_pnp;

  // create landmarks at the random locations
  landmark_map_sptr landmarks = kwiver::testing::init_landmarks(100);
  landmarks = kwiver::testing::noisy_landmarks(landmarks, 1.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq();

  // create tracks from the projections
  auto tracks = std::dynamic_pointer_cast<feature_track_set>(
    projected_tracks(landmarks, cameras));

  // add random noise to track image locations
  tracks = kwiver::testing::noisy_tracks(tracks, 0.5);

  const frame_id_t frame1 = 10;

  camera_map::map_camera_t cams = cameras->cameras();
  auto cam = std::dynamic_pointer_cast<camera_perspective>(cams[frame1]);
  camera_intrinsics_sptr cal = cam->intrinsics();

  // extract corresponding image points
  unsigned int i = 0;
  std::vector<vector_2d> pts_projs;
  std::vector<vector_3d> pts_3d;
  for (auto const& track : tracks->tracks())
  {
    auto lm_id = track->id();
    auto lm_it = landmarks->landmarks().find(lm_id);
    pts_3d.push_back(lm_it->second->loc());

    if (++i % 3 == 0)
    {
      pts_projs.push_back(kwiver::testing::random_point2d(1000.0));
    }
    else
    {
      auto const fts =
        std::dynamic_pointer_cast<feature_track_state>(*track->find(frame1));
      pts_projs.push_back(fts->feature->loc());
    }
  }

  // compute the camera pose from the points and their projections
  std::vector<bool> inliers;
  auto est_cam = est_pnp.estimate(pts_projs, pts_3d, cam->intrinsics(), inliers);

  // compare true and computed camera poses
  std::cout << "true R = " << cam->rotation().matrix() << std::endl;
  std::cout << "Estimated R = " << est_cam->rotation().matrix() << std::endl;
  std::cout << "true C = " << cam->center() << std::endl;
  std::cout << "Estimated C = " << est_cam->center() << std::endl;

  auto R_err = cam->rotation().inverse()*est_cam->rotation();
  std::cout << "Rotation error = " << R_err.angle()*180.0 / pi << " degrees" << std::endl;

  EXPECT_LT(R_err.angle(), outlier_rotation_tolerance);
  EXPECT_MATRIX_SIMILAR(cam->center(), est_cam->center(), outlier_center_tolerance);

  unsigned int num_inliers = std::count(inliers.begin(), inliers.end(), true);
  std::cout << "num inliers " << num_inliers << std::endl;
  EXPECT_GT(num_inliers, pts_projs.size() / 3)
    << "Not enough inliers";
}
