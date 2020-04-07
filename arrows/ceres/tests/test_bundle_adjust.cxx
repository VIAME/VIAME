/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * \brief test Ceres bundle adjustment functionality
 */

#include <test_eigen.h>
#include <test_scene.h>

#include <arrows/core/metrics.h>
#include <arrows/ceres/bundle_adjust.h>
#include <arrows/core/projected_track_set.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/math_constants.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;

using kwiver::arrows::ceres::bundle_adjust;

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

  EXPECT_NE( nullptr, algo::bundle_adjust::create("ceres") );
}

// ----------------------------------------------------------------------------
#include <arrows/tests/test_bundle_adjust.h>

// ----------------------------------------------------------------------------
// Add noise to landmarks and cameras and tracks before input to SBA; select
// a subset of tracks_states to make outliers (large observation noise); add a
// small amount of noise to all track states; and select a subset of
// tracks/track_states to constrain the problem
TEST(bundle_adjust, outlier_tracks)
{
  bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("max_num_iterations", 100);
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

  // make some observations outliers
  feature_track_set_sptr tracks_w_outliers =
      kwiver::testing::add_outliers_to_tracks(tracks, 0.1, 20.0);

  // remove some tracks/track_states and add Gaussian noise
  const double track_stdev = 1.0;
  feature_track_set_sptr tracks0 =
    kwiver::testing::noisy_tracks(
      kwiver::testing::subset_tracks(tracks_w_outliers, 0.5), track_stdev);


  double init_rmse = reprojection_rmse(cameras0->cameras(),
                                       landmarks0->landmarks(),
                                       tracks0->tracks());
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  double init_med_err = reprojection_median_error(cameras0->cameras(),
                                                  landmarks0->landmarks(),
                                                  tracks0->tracks());
  std::cout << "initial reprojection median error: "
            << init_med_err << std::endl;
  EXPECT_GE(init_med_err, 10.0)
    << "Initial reprojection median error should be large before SBA";

  // make a copy of the initial cameras and landmarks
  landmark_map_sptr landmarks1 =
    std::make_shared<simple_landmark_map>(landmarks0->landmarks());
  camera_map_sptr cameras1 =
    std::make_shared<simple_camera_map>(cameras0->cameras());

  // run bundle adjustement with the default, non-robust, trivial loss function
  ba.optimize(cameras0, landmarks0, tracks0);

  double trivial_loss_rmse = reprojection_rmse(cameras0->cameras(),
                                               landmarks0->landmarks(),
                                               tracks0->tracks());
  double trivial_loss_med_err = reprojection_median_error(cameras0->cameras(),
                                                          landmarks0->landmarks(),
                                                          tracks0->tracks());

  std::cout << "Non-robust SBA mean/median reprojection error: "
            << trivial_loss_rmse << "/" << trivial_loss_med_err << std::endl;
  EXPECT_GE( trivial_loss_med_err, track_stdev )
    << "Non-robust SBA should have a large median residual";

  // run bundle adjustment with a robust loss function
  cfg->set_value("loss_function_type", "HUBER_LOSS");
  ba.set_configuration(cfg);
  ba.optimize(cameras1, landmarks1, tracks0);

  double robust_loss_rmse = reprojection_rmse(cameras1->cameras(),
                                               landmarks1->landmarks(),
                                               tracks0->tracks());
  double robust_loss_med_err = reprojection_median_error(cameras1->cameras(),
                                                         landmarks1->landmarks(),
                                                         tracks0->tracks());

  std::cout << "Robust SBA mean/median reprojection error: "
            << robust_loss_rmse << "/" << robust_loss_med_err << std::endl;
  EXPECT_LE( trivial_loss_rmse, robust_loss_rmse )
    << "Robust SBA should increase RMSE error";
  EXPECT_GT( trivial_loss_med_err, robust_loss_med_err )
    << "Robust SBA should decrease median error";
  EXPECT_NEAR( robust_loss_med_err, 0.0, track_stdev );
}

// ----------------------------------------------------------------------------
// Helper for tests using distortion models in bundle adjustment
static void
test_ba_using_distortion( kwiver::vital::config_block_sptr cfg,
                          Eigen::VectorXd const& dc,
                          double estimate_tolerance = 0.0 )
{
  ceres::bundle_adjust ba;
  cfg->set_value("verbose", "true");
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640,480));
  K.set_dist_coeffs(dc);

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20,K);

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1);

  if ( estimate_tolerance != 0.0 )
  {
    // regenerate cameras without distortion so we can try to recover it
    K.set_dist_coeffs(Eigen::VectorXd());
    cameras = kwiver::testing::camera_seq(20,K);
  }

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
  EXPECT_NEAR( 0.0, end_rmse, 1e-5);

  // compare actual to estimated distortion parameters
  if ( estimate_tolerance != 0.0 )
  {
    auto cam0_ptr =
      std::dynamic_pointer_cast<camera_perspective>(cameras0->cameras()[0]);
    auto vdc2 = cam0_ptr->intrinsics()->dist_coeffs();
    // The estimated parameter vector can be longer and zero padded; lop off
    // any additional trailing values
    ASSERT_GE( vdc2.size(), static_cast<size_t>(dc.size()) );
    Eigen::VectorXd dc2{ Eigen::Map<Eigen::VectorXd>{ &vdc2[0], dc.size() } };

    Eigen::VectorXd diff = ( dc2 - dc ).cwiseAbs();
    std::cout << "distortion parameters\n"
              << "  actual:   " << dc.transpose() << "\n"
              << "  estimated: " << dc2.transpose() << "\n"
              << "  difference: " << diff.transpose() << std::endl;
    EXPECT_MATRIX_NEAR( dc, dc2, estimate_tolerance );
  }
}

// ----------------------------------------------------------------------------
static Eigen::VectorXd distortion_coefficients( int k )
{
  Eigen::VectorXd dc;
  switch (k)
  {
    case 1:
      dc.resize( 1 );
      dc << -0.01;
      return dc;

    case 2:
      dc.resize( 2 );
      dc << -0.01, 0.002;
      return dc;

    case 3:
      dc.resize( 5 );
      dc << -0.01, 0.002, 0, 0, -0.005;
      return dc;

    case 5:
      dc.resize( 5 );
      dc << -0.01, 0.002, -0.0005, 0.001, -0.005;
      return dc;

    case 8:
      dc.resize( 8 );
      dc << -0.01, 0.02, -0.0005, 0.001, 0.01, 0.02, 0.0007, -0.003;
      return dc;

    default:
      throw std::range_error{ "Invalid number of coefficients" };
  }
}

// ----------------------------------------------------------------------------
static char const* distortion_type( int k )
{
  switch (k)
  {
    case 1:
    case 2:
      return "POLYNOMIAL_RADIAL_DISTORTION";
    case 3:
    case 5:
      return "POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION";
    case 8:
      return "RATIONAL_RADIAL_TANGENTIAL_DISTORTION";
    default:
      throw std::range_error{ "Invalid number of coefficients" };
  }
}

// ----------------------------------------------------------------------------
static double distortion_estimation_tolerance( int k )
{
  switch (k)
  {
    case 1:
      return 1e-7;
    case 2:
      return 1e-6;
    case 3:
    case 5:
      return 1e-5;
    case 8:
      return 1e-2;
    default:
      throw std::range_error{ "Invalid number of coefficients" };
  }
}

// ----------------------------------------------------------------------------
class bundle_adjust_with_lens_distortion : public ::testing::TestWithParam<int>
{
};

// ----------------------------------------------------------------------------
TEST_P(bundle_adjust_with_lens_distortion, use_coefficients)
{
  auto const k = GetParam();
  auto const& dc = distortion_coefficients( k );

  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value( "lens_distortion_type", distortion_type( k ) );
  cfg->set_value( "optimize_dist_k1", false );
  cfg->set_value( "optimize_dist_k2", false );
  if ( k > 2 )
  {
    cfg->set_value( "optimize_dist_k3", false );
    cfg->set_value( "optimize_dist_p1_p2", false );
    if ( k > 5 )
    {
      cfg->set_value("optimize_dist_k4_k5_k6", false );
    }
  }

  test_ba_using_distortion( cfg, dc );
}

// ----------------------------------------------------------------------------
TEST_P(bundle_adjust_with_lens_distortion, estimate_coefficients)
{
  auto const k = GetParam();
  auto const& dc = distortion_coefficients( k );

  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value( "lens_distortion_type", distortion_type( k ) );
  cfg->set_value( "optimize_dist_k1", true );
  cfg->set_value( "optimize_dist_k2", ( k > 1 ) );
  if ( k > 2 )
  {
    cfg->set_value( "optimize_dist_k3", true );
    cfg->set_value( "optimize_dist_p1_p2", ( k > 3 ) );
    if ( k > 5 )
    {
      cfg->set_value("optimize_dist_k4_k5_k6", true );
    }
  }

  test_ba_using_distortion( cfg, dc, distortion_estimation_tolerance( k ) );
}

// ----------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(, bundle_adjust_with_lens_distortion,
                        ::testing::Values(1, 2, 3, 5, 8));

// ----------------------------------------------------------------------------
// Helper for tests of intrinsics sharing models in bundle adjustment; returns
// the number of unique camera intrinsics objects in the optimized cameras
static unsigned int
test_ba_intrinsic_sharing( camera_map_sptr cameras,
                           kwiver::vital::config_block_sptr cfg )
{
  ceres::bundle_adjust ba;
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0);

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
  EXPECT_GE( init_rmse, 10.0 )
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
                                      landmarks0->landmarks(),
                                      tracks->tracks());
  EXPECT_NEAR( 0.0, end_rmse, 1e-5 );

  std::set<camera_intrinsics_sptr> intrin_set;
  for ( auto const& ci : cameras0->cameras() )
  {
    auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>( ci.second );
    intrin_set.insert(cam_ptr->intrinsics());
  }

  return static_cast<unsigned int>(intrin_set.size());
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with forcing unique intrinsics
TEST(bundle_adjust, unique_intrinsics)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("camera_intrinsic_share_type", "FORCE_UNIQUE_INTRINSICS");

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640,480));

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20, K);
  EXPECT_EQ( cameras->size(), test_ba_intrinsic_sharing( cameras, cfg ) )
    << "Resulting camera intrinsics should be unique";
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with forcing common intrinsics
TEST(bundle_adjust, common_intrinsics)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("camera_intrinsic_share_type", "FORCE_COMMON_INTRINSICS");

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640,480));

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20, K);
  EXPECT_EQ( 1, test_ba_intrinsic_sharing( cameras, cfg ) )
    << "Resulting camera intrinsics should be unique";
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with multiple shared intrinics models
TEST(bundle_adjust, auto_share_intrinsics)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K1(1000, vector_2d(640,480));
  simple_camera_intrinsics K2(800, vector_2d(640,480));

  // create two camera sequences (elliptical paths)
  camera_map_sptr cameras1 = kwiver::testing::camera_seq(13, K1);
  camera_map_sptr cameras2 = kwiver::testing::camera_seq(7, K2);

  // combine the camera maps and offset the frame numbers
  const unsigned int offset = static_cast<unsigned int>(cameras1->size());
  camera_map::map_camera_t cams = cameras1->cameras();
  for ( auto const& ci : cameras2->cameras() )
  {
    cams[ci.first + offset] = ci.second;
  }

  auto cameras = std::make_shared<simple_camera_map>( cams );
  EXPECT_EQ( 2, test_ba_intrinsic_sharing( cameras, cfg ) )
    << "Resulting camera intrinsics should be unique";
}

// ----------------------------------------------------------------------------
// Helper for tests of different data scales
static void
test_ba_data_scales(kwiver::vital::config_block_sptr cfg,
                    double scale = 1.0)
{
  ceres::bundle_adjust ba;
  ba.set_configuration(cfg);

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640, 480));

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20, K, scale);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0 * scale);

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1*scale);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1*scale, 0.1);


  double init_rmse = reprojection_rmse(cameras0->cameras(),
    landmarks0->landmarks(),
    tracks->tracks());
  std::cout << "Data scaled by " << scale << "X" << std::endl;
  std::cout << "initial reprojection RMSE: " << init_rmse << std::endl;
  EXPECT_GE(init_rmse, 10.0)
    << "Initial reprojection RMSE should be large before SBA";

  ba.optimize(cameras0, landmarks0, tracks);

  double end_rmse = reprojection_rmse(cameras0->cameras(),
    landmarks0->landmarks(),
    tracks->tracks());
  std::cout << "Final reprojection RMSE: " << end_rmse << std::endl;
  EXPECT_NEAR(0.0, end_rmse, 1e-5);
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with different data scales
TEST(bundle_adjust, data_scales)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("camera_intrinsic_share_type", "FORCE_COMMON_INTRINSICS");

  test_ba_data_scales(cfg, 1.0);
  test_ba_data_scales(cfg, 10.0);
  test_ba_data_scales(cfg, 100.0);
  test_ba_data_scales(cfg, 1000.0);
}

// ----------------------------------------------------------------------------
// Helper for tests of camera smoothness constraints
static void
test_ba_camera_smoothing(camera_map_sptr cameras,
                         kwiver::vital::config_block_sptr cfg,
                         double scale = 1.0)
{
  ceres::bundle_adjust ba;
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0 * scale);

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1*scale);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1*scale, 0.1);


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
  std::cout << "Final reprojection RMSE: " << end_rmse << std::endl;
  EXPECT_NEAR(0.0, end_rmse, 0.1);
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with camera path smoothness
TEST(bundle_adjust, camera_path_smoothness)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("camera_intrinsic_share_type", "FORCE_COMMON_INTRINSICS");
  cfg->set_value("camera_path_smoothness", 1.0);

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640, 480));

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20, K);
  test_ba_camera_smoothing(cameras, cfg);

  // test cameras at a larger scale
  cameras = kwiver::testing::camera_seq(20, K, 1000.0);
  test_ba_camera_smoothing(cameras, cfg, 1000.0);

  // create a camera sequence (elliptical path)
  cameras = kwiver::testing::camera_seq(100, K);
  test_ba_camera_smoothing(cameras, cfg);

  // test with non-sequential cameras
  auto cams = cameras->cameras();
  for (auto frame : { 2, 3, 6, 11, 13, 19, 20, 21, 23,
                      24, 27, 33, 34, 50, 51, 53 })
  {
    cams.erase(frame);
  }
  cameras = std::make_shared<simple_camera_map>(cams);
  test_ba_camera_smoothing(cameras, cfg);
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with camera forward motion damping
TEST(bundle_adjust, camera_forward_motion_damping)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  // forward motion damping only applies to unique intrinsics
  cfg->set_value("camera_intrinsic_share_type", "FORCE_UNIQUE_INTRINSICS");
  cfg->set_value("camera_forward_motion_damping", 0.1);

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640, 480));

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20, K);
  test_ba_camera_smoothing(cameras, cfg);

  // test cameras at a larger scale
  cameras = kwiver::testing::camera_seq(20, K, 1000.0);
  test_ba_camera_smoothing(cameras, cfg, 1000.0);

  // create a camera sequence (elliptical path)
  cameras = kwiver::testing::camera_seq(100, K);
  test_ba_camera_smoothing(cameras, cfg);

  // test with non-sequential cameras
  auto cams = cameras->cameras();
  for (auto frame : { 2, 3, 6, 11, 13, 19, 20, 21, 23,
                      24, 27, 33, 34, 50, 51, 53 })
  {
    cams.erase(frame);
  }
  cameras = std::make_shared<simple_camera_map>(cams);
  test_ba_camera_smoothing(cameras, cfg);
}

// ----------------------------------------------------------------------------
// Helper for tests of hfov constraints
static void
test_ba_min_hfov(camera_map_sptr cameras,
  kwiver::vital::config_block_sptr cfg,
  double scale = 1.0)
{
  ceres::bundle_adjust ba;
  ba.set_configuration(cfg);

  // create landmarks at the corners of a cube
  landmark_map_sptr landmarks = kwiver::testing::cube_corners(2.0 * scale);

  // create tracks from the projections
  feature_track_set_sptr tracks = projected_tracks(landmarks, cameras);

  // add Gaussian noise to the landmark positions
  landmark_map_sptr landmarks0 = kwiver::testing::noisy_landmarks(landmarks, 0.1*scale);

  // add Gaussian noise to the camera positions and orientations
  camera_map_sptr cameras0 = kwiver::testing::noisy_cameras(cameras, 0.1*scale, 0.1);


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
  std::cout << "Final reprojection RMSE: " << end_rmse << std::endl;
  EXPECT_NEAR(0.0, end_rmse, 2.0);

  auto cam = std::static_pointer_cast<kwiver::vital::camera_perspective>(
    cameras0->cameras().begin()->second);
  double f = cam->intrinsics()->focal_length();
  double half_w = cam->intrinsics()->principal_point()[0];
  double hfov = std::atan(half_w / f) * 2 * kwiver::vital::rad_to_deg;
  std::cout << "Final horizontal FOV: " << hfov << std::endl;
  // allow one degree of tolerance because minimum_hfov is a soft limit
  EXPECT_GE(hfov, cfg->get_value<double>("minimum_hfov")-1.0)
    << "estimated H-FOV should not be less than minimum";
}

// ----------------------------------------------------------------------------
// Test bundle adjustment with minimum horizontal FOV
TEST(bundle_adjust, minimum_hfov)
{
  ceres::bundle_adjust ba;
  config_block_sptr cfg = ba.get_configuration();
  cfg->set_value("verbose", "true");
  cfg->set_value("camera_intrinsic_share_type", "FORCE_COMMON_INTRINSICS");
  cfg->set_value("minimum_hfov", 70);

  // The intrinsic camera parameters to use
  simple_camera_intrinsics K(1000, vector_2d(640, 480));

  // create a camera sequence (elliptical path)
  camera_map_sptr cameras = kwiver::testing::camera_seq(20, K);
  test_ba_min_hfov(cameras, cfg);


  // create a camera sequence (elliptical path)
  cameras = kwiver::testing::camera_seq(100, K);
  test_ba_min_hfov(cameras, cfg);


  // create a camera sequence (elliptical path)
  cameras = kwiver::testing::camera_seq(100, K, 1000.0);
  test_ba_min_hfov(cameras, cfg, 1000.0);

  // test with non-sequential cameras
  auto cams = cameras->cameras();
  for (auto frame : { 2, 3, 6, 11, 13, 19, 20, 21, 23,
    24, 27, 33, 34, 50, 51, 53 })
  {
    cams.erase(frame);
  }
  cameras = std::make_shared<simple_camera_map>(cams);
  test_ba_min_hfov(cameras, cfg, 1000.0);
}
