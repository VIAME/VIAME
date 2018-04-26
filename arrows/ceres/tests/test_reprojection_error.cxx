/*ckwg +29
 * Copyright 2015-2018 by Kitware, Inc.
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
 * \brief test Ceres reprojection error functors
 */

#include <test_scene.h>

#include <arrows/ceres/reprojection_error.h>
#include <arrows/ceres/types.h>

#include <arrows/core/metrics.h>
#include <arrows/core/projected_track_set.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

using kwiver::arrows::reprojection_rmse;
using kwiver::arrows::projected_tracks;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
/// Test the reprojection error of a single residual
static void
test_reprojection_error(
  camera_perspective const& cam, landmark const& lm, feature const& f,
  kwiver::arrows::ceres::LensDistortionType dist_type )
{
  ::ceres::CostFunction* cost_func =
      create_cost_func(dist_type, f.loc().x(), f.loc().y());

  double pose[6];
  vector_3d rot = cam.rotation().rodrigues();
  std::copy(rot.data(), rot.data()+3, pose);
  vector_3d center = cam.center();
  std::copy(center.data(), center.data()+3, pose+3);

  unsigned int ndp = num_distortion_params(dist_type);
  std::vector<double> intrinsics(5+ndp, 0.0);
  camera_intrinsics_sptr K = cam.intrinsics();
  intrinsics[0] = K->focal_length();
  intrinsics[1] = K->principal_point().x();
  intrinsics[2] = K->principal_point().y();
  intrinsics[3] = K->aspect_ratio();
  intrinsics[4] = K->skew();
  const std::vector<double> d = K->dist_coeffs();
  // copy the intersection of parameters provided in K
  // and those that are supported by the requested model type
  unsigned int num_dp = std::min(ndp, static_cast<unsigned int>(d.size()));
  if( num_dp > 0 )
  {
    std::copy(d.begin(), d.begin()+num_dp, &intrinsics[5]);
  }

  double point[3] = {lm.loc().x(), lm.loc().y(), lm.loc().z()};

  double* parameters[3] = {&intrinsics[0], pose, point};
  vector_2d residuals;
  cost_func->Evaluate(parameters, residuals.data(), NULL);
  delete cost_func;

  EXPECT_NEAR( 0.0, residuals.norm(), 1e-12 );
}

// ----------------------------------------------------------------------------
static Eigen::VectorXd distortion_coefficients( int dim )
{
  Eigen::VectorXd dc{ 8 };
  dc << -0.01, 0.002, 0.001, -0.005, -0.004, 0.02, -0.007, 0.0001;
  dc.conservativeResize( dim );
  return dc;
}

// ----------------------------------------------------------------------------
struct reprojection_test
{
  char const* const distortion_model;
  kwiver::arrows::ceres::LensDistortionType const distortion_type;
  int const distortion_coefficients_dimension;
};

// ----------------------------------------------------------------------------
void
PrintTo( reprojection_test const& v, ::std::ostream* os )
{
  (*os) << v.distortion_model << '/' << v.distortion_coefficients_dimension;
}

// ----------------------------------------------------------------------------
class reprojection_error : public ::testing::TestWithParam<reprojection_test>
{
};

// ----------------------------------------------------------------------------
// Compare MAP-Tk camera projection to Ceres reprojection error models
TEST_P(reprojection_error, compare_projections)
{
  auto const dist_type = GetParam().distortion_type;

  // Create landmarks at the corners of a cube
  auto landmarks = kwiver::testing::cube_corners( 2.0 );

  // The intrinsic camera parameters to use
  auto const dc_dim = GetParam().distortion_coefficients_dimension;
  simple_camera_intrinsics K{ 1000, vector_2d{ 640, 480 } };
  K.set_dist_coeffs( distortion_coefficients( dc_dim ) );

  // Create a camera sequence (elliptical path)
  auto cameras = kwiver::testing::camera_seq( 20, K );

  // Create tracks from the projections
  auto tracks = projected_tracks( landmarks, cameras );

  // Test the reprojection error of all residuals
  auto cam_map = cameras->cameras();
  auto lm_map = landmarks->landmarks();
  auto trks = tracks->tracks();

  double rmse = reprojection_rmse(cam_map, lm_map, trks);
  std::cout << "MAP-Tk reprojection RMSE: " << rmse << std::endl;
  EXPECT_NEAR( 0.0, rmse, 1e-12 )
    << "MAP-Tk reprojection RMSE should be small";

  for ( track_sptr const& t : trks )
  {
    auto lmi = lm_map.find( t->id() );
    if ( lmi == lm_map.end() || !lmi->second )
    {
      // no landmark corresponding to this track
      continue;
    }

    SCOPED_TRACE( "At track " + std::to_string( t->id() ) );

    const landmark& lm = *lmi->second;
    for( auto const& ts : *t )
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>( ts );
      if ( !fts || !fts->feature )
      {
        // no feature for this track state.
        continue;
      }

      auto const& feat = *fts->feature;
      auto ci = cam_map.find( ts->frame() );
      if ( ci == cam_map.end() || !ci->second )
      {
        // no camera corresponding to this track state
        continue;
      }

      SCOPED_TRACE( "At track frame " + std::to_string( ts->frame() ) );

      auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>(ci->second);
      test_reprojection_error( *cam_ptr, lm, feat, dist_type );
    }
  }
}

// ----------------------------------------------------------------------------
#define DISTORTION( t, k ) \
  reprojection_test{ #t, kwiver::arrows::ceres::t, k }

INSTANTIATE_TEST_CASE_P(
  ,
  reprojection_error,
  ::testing::Values(
    DISTORTION( NO_DISTORTION, 0 ),
    DISTORTION( POLYNOMIAL_RADIAL_DISTORTION, 0 ),
    DISTORTION( POLYNOMIAL_RADIAL_DISTORTION, 1 ),
    DISTORTION( POLYNOMIAL_RADIAL_DISTORTION, 2 ),
    DISTORTION( POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION, 0 ),
    DISTORTION( POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION, 1 ),
    DISTORTION( POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION, 2 ),
    DISTORTION( POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION, 4 ),
    DISTORTION( POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION, 5 ),
    DISTORTION( RATIONAL_RADIAL_TANGENTIAL_DISTORTION, 0 ),
    DISTORTION( RATIONAL_RADIAL_TANGENTIAL_DISTORTION, 1 ),
    DISTORTION( RATIONAL_RADIAL_TANGENTIAL_DISTORTION, 2 ),
    DISTORTION( RATIONAL_RADIAL_TANGENTIAL_DISTORTION, 4 ),
    DISTORTION( RATIONAL_RADIAL_TANGENTIAL_DISTORTION, 5 ),
    DISTORTION( RATIONAL_RADIAL_TANGENTIAL_DISTORTION, 8 )
  ) );
