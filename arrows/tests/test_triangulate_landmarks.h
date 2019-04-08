/*ckwg +29
 * Copyright 2014-2019 by Kitware, Inc.
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
 *
 * \brief share tests for triangulate_landmarks implementations
 *
 */

#ifndef ALGORITHMS_TEST_ALGO_TEST_TRIANGULATE_LANDMARKS_H_
#define ALGORITHMS_TEST_ALGO_TEST_TRIANGULATE_LANDMARKS_H_

#include <test_scene.h>

#include <arrows/core/projected_track_set.h>
#include <arrows/core/metrics.h>
#include <vital/algo/triangulate_landmarks.h>

#include <gtest/gtest.h>

namespace kwiver {
namespace testing {

// ----------------------------------------------------------------------------
// Input to triangulation is the ideal solution, make sure it doesn't diverge
void test_from_solution(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  //+ using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  kwiver::vital::landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  kwiver::vital::camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  kwiver::vital::feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  double init_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                       landmarks->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;

  EXPECT_LE(init_rmse, 1e-12) << "Initial reprojection RMSE should be small";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras, tracks, landmarks);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                      landmarks->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-12) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
// Add noise to landmarks before input to triangulation
void test_noisy_landmarks(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = testing::noisy_landmarks(landmarks, 0.1);


  double init_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras, tracks, landmarks0);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
// Initialize all landmarks to the origin as input to triangulation
void test_zero_landmarks(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // initialize all landmarks to the origin
  landmark_id_t num_landmarks = static_cast<landmark_id_t>(landmarks->size());
  landmark_map_sptr landmarks0 = testing::init_landmarks(num_landmarks);


  double init_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras, tracks, landmarks0);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
// Select a subset of cameras to triangulation from
void test_subset_cameras(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // initialize all landmarks to the origin
  landmark_id_t num_landmarks = static_cast<landmark_id_t>(landmarks->size());
  landmark_map_sptr landmarks0 = testing::init_landmarks(num_landmarks);

  camera_map::map_camera_t cam_map = cameras->cameras();
  camera_map::map_camera_t cam_map2;
  for ( auto const& p : cam_map )
  {
    /// take every third camera
    if(p.first % 3 == 0)
    {
      cam_map2.insert(p);
    }
  }
  camera_map_sptr cameras0(new simple_camera_map(cam_map2));


  EXPECT_EQ(7, cameras0->size()) << "Reduced number of cameras";

  double init_rmse = kwiver::arrows::reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras0, tracks, landmarks0);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
// Select a subset of landmarks to triangulate
void test_subset_landmarks(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // initialize all landmarks to the origin
  landmark_id_t num_landmarks = static_cast<landmark_id_t>(landmarks->size());
  landmark_map_sptr landmarks0 = testing::init_landmarks(num_landmarks);

  // remove some landmarks
  landmark_map::map_landmark_t lm_map = landmarks0->landmarks();
  lm_map.erase(1);
  lm_map.erase(4);
  lm_map.erase(5);
  landmarks0 = landmark_map_sptr(new simple_landmark_map(lm_map));

  EXPECT_EQ(5, landmarks0->size()) << "Reduced number of landmarks";

  double init_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras, tracks, landmarks0);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
// Select a subset of tracks/track_states to constrain the problem
void test_subset_tracks(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // initialize all landmarks to the origin
  landmark_id_t num_landmarks = static_cast<landmark_id_t>(landmarks->size());
  landmark_map_sptr landmarks0 = testing::init_landmarks(num_landmarks);

  // remove some tracks/track_states
  feature_track_set_sptr tracks0 = testing::subset_tracks(tracks, 0.5);

  double init_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks0->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras, tracks0, landmarks0);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks0->tracks());
  EXPECT_NEAR(0.0, end_rmse, 1e-5) << "RMSE after triangulation";
}

// ----------------------------------------------------------------------------
// Select a subset of tracks/track_states and add noise
void test_noisy_tracks(kwiver::vital::algo::triangulate_landmarks& tri_lm)
{
  using namespace kwiver::vital;

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = testing::cube_corners(2.0);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = testing::camera_seq();

  // create tracks from the projections
  feature_track_set_sptr tracks = kwiver::arrows::projected_tracks(landmarks, cameras);

  // initialize all landmarks to the origin
  landmark_id_t num_landmarks = static_cast<landmark_id_t>(landmarks->size());
  landmark_map_sptr landmarks0 = testing::init_landmarks(num_landmarks);

  // remove some tracks/track_states and add Gaussian noise
  const double track_stdev = 1.0;
  feature_track_set_sptr tracks0 = testing::noisy_tracks(
                               testing::subset_tracks(tracks, 0.5),
                               track_stdev);


  double init_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                       landmarks0->landmarks(),
                                       tracks0->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before triangulation";

  reset_inlier_flag( tracks );
  tri_lm.triangulate(cameras, tracks0, landmarks0);

  double end_rmse = kwiver::arrows::reprojection_rmse(cameras->cameras(),
                                      landmarks0->landmarks(),
                                      tracks0->tracks());
  EXPECT_NEAR(0.0, end_rmse, 3.0*track_stdev) << "RMSE after triangulation";
}

} // end namespace testing
} // end namespace kwiver

#endif // ALGORITHMS_TEST_ALGO_TEST_TRIANGULATE_LANDMARKS_H_
