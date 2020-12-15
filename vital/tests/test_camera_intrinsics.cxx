// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core camera_intrinsics tests
 */

#include <test_eigen.h>

#include <vital/types/camera_intrinsics.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(camera_intrinsics, map)
{
  vector_2d pp{ 300, 400 };
  double f = 1000.0;
  double a = 0.75;
  double s = 0.1;
  simple_camera_intrinsics K{ f, pp, a, s };

  EXPECT_MATRIX_NEAR( pp, K.map( vector_2d{ 0, 0 } ), 1e-12 );
  EXPECT_MATRIX_NEAR( ( vector_2d{ 0, 0 } ), K.unmap( pp ), 1e-12 );

  vector_2d test_pt{ 1, 2 };
  vector_2d mapped_test_gt{ test_pt.x() * f + test_pt.y() * s + pp.x(),
                            test_pt.y() * f / a + pp.y() };
  EXPECT_MATRIX_NEAR( mapped_test_gt, K.map( test_pt ), 1e-12 );
  EXPECT_MATRIX_NEAR( test_pt, K.unmap( K.map( test_pt ) ), 1e-12 );

  vector_3d homg_pt{ 2.5 * vector_3d{ test_pt.x(), test_pt.y(), 1 } };
  EXPECT_MATRIX_NEAR( mapped_test_gt, K.map( homg_pt ), 1e-12 );
}

// ----------------------------------------------------------------------------
double dist_deriv(double r, double a, double b, double c)
{
  double r2 = r*r;
  return 1 + 3 * a*r2 + 5 * b*r2*r2 + 7 * c*r2*r2*r2;
}

void test_finite_max_radius(double a, double b, double c)
{
  double mr = simple_camera_intrinsics::max_distort_radius_sq(a, b, c);
  EXPECT_GT(mr, 0.0);
  EXPECT_TRUE(std::isfinite(mr));
  EXPECT_NEAR(dist_deriv(std::sqrt(mr), a, b, c), 0.0, 1e-12);
}

void test_infinite_max_radius(double a, double b, double c)
{
  double mr = simple_camera_intrinsics::max_distort_radius_sq(a, b, c);
  EXPECT_TRUE(mr == std::numeric_limits<double>::infinity());
}

TEST(camera_intrinsics, max_radius)
{
  // parameters that should have infinite radius
  test_infinite_max_radius(0.0, 0.0, 0.0);
  test_infinite_max_radius(0.1, 0.2, 0.3);
  test_infinite_max_radius(0.1, 0.2, 0.0);
  test_infinite_max_radius(0.1, 0.0, 0.0);
  test_infinite_max_radius(-0.1, 0.005, 0.00001);

  // these parameters have three solutions
  test_finite_max_radius(-0.1, 0.004, -0.00001);
  // these parameters have two solutions
  test_finite_max_radius(-0.1, 0.001, 0.00001);
  // these parameters have one solution
  test_finite_max_radius(-0.01, -0.01, -0.0001);

  // these parameters have three identical solutions (all 2.0)
  test_finite_max_radius(-0.5, 0.15, -1.0/56);
  // these parameters have two identical solutions (1, 1, 2)
  test_finite_max_radius(-5.0/6, 2.0/5, -1.0 / 14);
  // these parameters have two identical solutions (1, 1)
  test_finite_max_radius(-2.0/3, 1.0/5, 0.0);

  test_finite_max_radius(-0.1, 0.0, 0.0);
  test_finite_max_radius(0.0, -0.2, 0.0);
  test_finite_max_radius(0.0, 0.0, -0.3);
}

// ----------------------------------------------------------------------------
TEST(camera_intrinsics, is_map_valid)
{
  vector_2d pp{ 300, 400 };
  double f = 1000.0;
  double a = 1.0;
  double s = 0.0;
  vector_3d d = { -0.1, -0.01, 0.001 };
  // max radius for this d is about 1.61618
  simple_camera_intrinsics K{ f, pp, a, s, d };

  EXPECT_TRUE(K.is_map_valid(vector_2d(0.5, 0.4)));
  EXPECT_TRUE(K.is_map_valid(vector_2d(1.0, -1.0)));
  EXPECT_TRUE(K.is_map_valid(vector_3d(-2.0, 1.0, 2.0)));
  EXPECT_FALSE(K.is_map_valid(vector_2d(2.0, 1.0)));
  EXPECT_FALSE(K.is_map_valid(vector_2d(-100.0, 200.0)));
  // this point is at infinity, so it's always invalid
  EXPECT_FALSE(K.is_map_valid(vector_3d(2.0, 1.0, 0.0)));

  d = { 0.1, 0.01, 0.001 };
  // max radius for this d is infinity
  K.set_dist_coeffs(d);

  EXPECT_TRUE(K.is_map_valid(vector_2d(0.5, 0.4)));
  EXPECT_TRUE(K.is_map_valid(vector_2d(1.0, -1.0)));
  EXPECT_TRUE(K.is_map_valid(vector_3d(-2.0, 1.0, 2.0)));
  EXPECT_TRUE(K.is_map_valid(vector_2d(2.0, 1.0)));
  EXPECT_TRUE(K.is_map_valid(vector_2d(-100.0, 200.0)));
  // this point is at infinity, so it's always invalid
  EXPECT_FALSE(K.is_map_valid(vector_3d(2.0, 1.0, 0.0)));
}
