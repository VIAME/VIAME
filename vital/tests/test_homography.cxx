// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Homography and derived class functionality regression tests
 */

#include <test_eigen.h>

#include <vital/exceptions/math.h>
#include <vital/types/homography.h>
#include <vital/types/homography_f2f.h>

#include <Eigen/LU>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
template <typename T>
class homography : public ::testing::Test
{
};

using types = ::testing::Types<float, double>;
TYPED_TEST_CASE(homography, types);

// ----------------------------------------------------------------------------
// Test invert function Eigen::Matrix derived classes
TYPED_TEST(homography, inversion)
{
  using homography_t = typename kwiver::vital::homography_<TypeParam>;

  homography_t invertible;
  homography_t expected_result;
  homography_t noninvertable;
  homography_t ni_result;

  invertible.get_matrix() << 1, 1, 2,
                             3, 4, 5,
                             6, 7, 9;
  expected_result.get_matrix() << -0.5, -2.5,  1.5,
                                  -1.5,  1.5, -0.5,
                                   1.5,  0.5, -0.5;

  homography_t h_inverse( *invertible.inverse() );
  EXPECT_MATRIX_EQ( expected_result.matrix(), h_inverse.matrix() );

  noninvertable.get_matrix() << 1, 2, 3,
                                4, 5, 6,
                                7, 8, 9;
  bool is_invertible;
  noninvertable.get_matrix().computeInverseWithCheck(
    ni_result.get_matrix(), is_invertible );
  EXPECT_FALSE( is_invertible );
}

// ----------------------------------------------------------------------------
// Test mapping a point for a homography/point data type
TYPED_TEST(homography, map_point_zero_div)
{
  using homography_t = typename kwiver::vital::homography_<TypeParam>;
  using vector_t = typename Eigen::Matrix<TypeParam, 2, 1>;

  vector_t test_p{ 1, 1 };
  TypeParam e = Eigen::NumTraits<TypeParam>::dummy_precision();

  // Where [2,2] = 0
  homography_t h_0;
  h_0.get_matrix() << 1.0, 0.0, 1.0,
                      0.0, 1.0, 1.0,
                      0.0, 0.0, 0.0;
  EXPECT_THROW(
    h_0.map_point( test_p ),
    kwiver::vital::point_maps_to_infinity )
    << "Applying point to matrix with 0-value lower-right corner";

  // Where [2,2] = e, which is the approximately-zero threshold
  homography_t h_e;
  h_e.get_matrix() << 1.0, 0.0, 1.0,
                      0.0, 1.0, 1.0,
                      0.0, 0.0,  e ;
  EXPECT_THROW(
    h_e.map_point( test_p ),
    kwiver::vital::point_maps_to_infinity )
    << "Applying point to matrix with e-value lower-right corner";

  // Where [2,2] = 0.5, which should be valid.
  homography_t h_half;
  h_half.get_matrix() << 1.0, 0.0, 1.0,
                         0.0, 1.0, 1.0,
                         0.0, 0.0, 0.5;
  EXPECT_MATRIX_NEAR( ( vector_t{ 4, 4 } ), h_half.map_point( test_p ), e );
}

// ----------------------------------------------------------------------------
TYPED_TEST(homography, map_point)
{
  using homography_t = typename kwiver::vital::homography_<TypeParam>;
  using vector_t = typename Eigen::Matrix<TypeParam, 2, 1>;

  // Identity transformation
  homography_t h;
  vector_t p{ static_cast<TypeParam>( 2.2 ), static_cast<TypeParam>( 5.5 ) };
  EXPECT_EQ( p, h.map_point( p ) );
}

// ----------------------------------------------------------------------------
TEST(f2f_homography, inversion)
{
  // Testing from and to frame swapping during inversion
  kwiver::vital::matrix_3x3d i{ kwiver::vital::matrix_3x3d::Identity() };
  kwiver::vital::f2f_homography h{ i, 0, 10 };
  kwiver::vital::f2f_homography h_inv = h.inverse();

  EXPECT_EQ( 10, h_inv.from_id() );
  EXPECT_EQ( 0, h_inv.to_id() );
}
