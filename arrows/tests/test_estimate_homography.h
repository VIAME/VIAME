// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test homography estimation algorithm
 */

#include <test_eigen.h>
#include <test_random_point.h>

#include <vital/plugin_loader/plugin_manager.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;
using namespace kwiver::testing;

// ----------------------------------------------------------------------------
TEST(estimate_homography, not_enough_points)
{
  estimate_homography estimator;

  std::vector<vector_2d> pts1, pts2;
  std::vector<bool> inliers;

  homography_sptr H = estimator.estimate(pts1, pts2, inliers);
  EXPECT_EQ( nullptr, H )
    << "Estimation with no points should produce a NULL homography";

  pts1.push_back(vector_2d(0.0, 0.0));
  pts1.push_back(vector_2d(2.0, 0.0));
  pts1.push_back(vector_2d(0.0, 2.0));

  pts2.push_back(vector_2d(1.0, 1.0));
  pts2.push_back(vector_2d(4.0, 1.0));
  pts2.push_back(vector_2d(1.0, 4.0));

  H = estimator.estimate(pts1, pts2, inliers);
  EXPECT_EQ( nullptr, H )
    << "Estimation with < 4 points should produce a NULL homography";
}

// ----------------------------------------------------------------------------
static kwiver::vital::matrix_3x3d sample_homography()
{
  kwiver::vital::matrix_3x3d H;
  H << 2.0, 0.0, -1.5,
       0.0, 3.0, 5.0,
       0.0, 0.0, 1.0;
  return H;
}

// ----------------------------------------------------------------------------
TEST(estimate_homography, four_points)
{
  estimate_homography estimator;

  std::vector<vector_2d> pts1, pts2;
  std::vector<bool> inliers;

  pts1.push_back(vector_2d(0.0, 0.0));
  pts1.push_back(vector_2d(2.0, 0.0));
  pts1.push_back(vector_2d(0.0, 2.0));
  pts1.push_back(vector_2d(2.0, 2.0));

  matrix_3x3d true_H = sample_homography();

  // transform pts1 to pts2 using true_H
  for(unsigned i=0; i<pts1.size(); ++i)
  {
    vector_3d v = true_H * vector_3d(pts1[i].x(), pts1[i].y(), 1.0);
    pts2.push_back(vector_2d(v.x()/v.z(), v.y()/v.z()));
  }

  homography_sptr estimated_H = estimator.estimate(pts1, pts2, inliers);
  estimated_H = estimated_H->normalize();

  double H_error = (true_H - estimated_H->matrix()).norm();
  std::cout << "Homography estimation error: "<< H_error << std::endl;

  EXPECT_MATRIX_NEAR( true_H, estimated_H->matrix(), 1e-14 );
  EXPECT_LT( H_error, 1e-14 );
}

// ----------------------------------------------------------------------------
TEST(estimate_homography, ideal_points)
{
  estimate_homography estimator;

  matrix_3x3d true_H = sample_homography();

  // create random points that perfectly correspond via true_H
  std::vector<vector_2d> pts1, pts2;
  for(unsigned i=0; i<100; ++i)
  {
    vector_2d v2 = random_point2d(1000.0) + vector_2d(500.0,500.0);
    pts1.push_back(v2);
    vector_3d v3 = true_H * vector_3d(v2.x(), v2.y(), 1.0);
    pts2.push_back(vector_2d(v3.x()/v3.z(), v3.y()/v3.z()));
  }

  std::vector<bool> inliers;
  homography_sptr estimated_H = estimator.estimate(pts1, pts2, inliers);
  estimated_H = estimated_H->normalize();

  double H_error = (true_H - estimated_H->matrix()).norm();
  std::cout << "Homography estimation error: "<< H_error << std::endl;
  EXPECT_MATRIX_NEAR( true_H, estimated_H->matrix(), ideal_matrix_tolerance );
  EXPECT_LT( H_error, ideal_norm_tolerance );

  std::cout << "num inliers " << inliers.size() << std::endl;
  EXPECT_EQ( 100, inliers.size() ) << "All points should be inliers";
}

// ----------------------------------------------------------------------------
TEST(estimate_homography, noisy_points)
{
  estimate_homography estimator;

  matrix_3x3d true_H = sample_homography();

  // create random points + noise that approximately correspond via true_H
  std::vector<vector_2d> pts1, pts2;
  for(unsigned i=0; i<100; ++i)
  {
    vector_2d v2 = random_point2d(1000.0) + vector_2d(500.0,500.0);
    pts1.push_back(v2 + random_point2d(0.1));
    vector_3d v3 = true_H * vector_3d(v2.x(), v2.y(), 1.0);
    pts2.push_back(vector_2d(v3.x()/v3.z(), v3.y()/v3.z())
                   + random_point2d(0.1));
  }

  std::vector<bool> inliers;
  homography_sptr estimated_H = estimator.estimate(pts1, pts2, inliers);
  estimated_H = estimated_H->normalize();

  double H_error = (true_H - estimated_H->matrix()).norm();
  std::cout << "Homography estimation error: "<< H_error << std::endl;
  EXPECT_MATRIX_NEAR( true_H, estimated_H->matrix(), noisy_matrix_tolerance );
  EXPECT_LT( H_error, noisy_norm_tolerance );

  std::cout << "num inliers " << inliers.size() << std::endl;
  EXPECT_GE( inliers.size(), 90 )
    << "At least 90% of points should be inliers";
}

// ----------------------------------------------------------------------------
TEST(estimate_homography, outlier_points)
{
  estimate_homography estimator;

  matrix_3x3d true_H = sample_homography();

  // create random points + noise that approximately correspond via true_H
  std::vector<vector_2d> pts1, pts2;
  std::vector<bool> true_inliers;
  for(unsigned i=0; i<100; ++i)
  {
    vector_2d v2 = random_point2d(1000.0) + vector_2d(500.0,500.0);
    pts1.push_back(v2);
    vector_3d v3 = true_H * vector_3d(v2.x(), v2.y(), 1.0);
    pts2.push_back(vector_2d(v3.x()/v3.z(), v3.y()/v3.z()));
    true_inliers.push_back(true);
    if (i%3 == 0)
    {
      pts2.back() = random_point2d(1000.0) + vector_2d(500.0,500.0);
      true_inliers.back() = false;
    }
  }

  std::vector<bool> inliers;
  homography_sptr estimated_H = estimator.estimate(pts1, pts2, inliers);
  estimated_H = estimated_H->normalize();

  double H_error = (true_H - estimated_H->matrix()).norm();
  std::cout << "Homography estimation error: "<< H_error << std::endl;
  EXPECT_MATRIX_NEAR( true_H, estimated_H->matrix(), outlier_matrix_tolerance );
  EXPECT_LT( H_error, outlier_norm_tolerance );

  std::cout << "num inliers " << inliers.size() << std::endl;

  unsigned correct_inliers = 0;
  for( unsigned i=0; i<inliers.size(); ++i )
  {
    if (true_inliers[i] == inliers[i])
    {
      ++correct_inliers;
    }
  }
  std::cout << "correct inliers " << correct_inliers << std::endl;
  EXPECT_GE( correct_inliers, 90 )
    << "At least 90% of points should be correct inliers";
}
