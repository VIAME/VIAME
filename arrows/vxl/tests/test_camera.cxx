// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL camera class functionality
 */

#include <test_eigen.h>

#include <arrows/vxl/camera.h>

#include <Eigen/QR>

#include <cstdio>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
template <typename T>
static
vpgl_perspective_camera<T> sample_vpgl_camera()
{
  using namespace kwiver::arrows;
  vpgl_calibration_matrix<T> vk(T(4000), vgl_point_2d<T>(300,400),
                                T(1.0), T(0.75), T(0.0001));
  vgl_rotation_3d<T> vr(T(0.7), T(0.1), T(1.3));
  vgl_point_3d<T> vc(T(100.0), T(0.0), T(20.0));
  return vpgl_perspective_camera<T>(vk, vc, vr);
}

// ----------------------------------------------------------------------------
template <typename T, unsigned U, unsigned V>
static
Eigen::Matrix<T, U, V> as_eigen( vnl_matrix_fixed<T, U, V> const& in )
{
  Eigen::Matrix<T, U, V> out;
  for ( unsigned u = 0; u < U; ++u )
  {
    for ( unsigned v = 0; v < V; ++v )
    {
      out(u, v) = in(u, v);
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
template <typename T>
static
Eigen::Quaternion<T> as_eigen( vnl_quaternion<T> const& in )
{
  return { in.r(), in.x(), in.y(), in.z() };
}

// ----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, 3, 1> as_eigen( vgl_point_3d<T> const& in )
{
  return { in.x(), in.y(), in.z() };
}

// ----------------------------------------------------------------------------
TEST(camera, convert_camera_sptr)
{
  using namespace kwiver::arrows;
  vpgl_perspective_camera<double> vcd = sample_vpgl_camera<double>();
  vpgl_perspective_camera<float> vcf = sample_vpgl_camera<float>();

  EXPECT_NE( nullptr, vxl::vpgl_camera_to_vital(vcd) )
    << "Type conversion from double camera";

  EXPECT_NE( nullptr, vxl::vpgl_camera_to_vital(vcf) )
    << "Type conversion from float camera";
}

// ----------------------------------------------------------------------------
template <typename T>
void test_convert_camera(T eps)
{
  using namespace kwiver::arrows;
  vpgl_perspective_camera<T> vcam = sample_vpgl_camera<T>();

  simple_camera_perspective mcam;
  vxl::vpgl_camera_to_vital(vcam, mcam);

  std::cerr << "rotation: " << mcam.rotation().quaternion().coeffs() << std::endl;

  vpgl_perspective_camera<T> vcam2;
  vxl::vital_to_vpgl_camera(mcam, vcam2);

  auto const& calibration = as_eigen( vcam.get_calibration().get_matrix() );
  auto const& calibration2 = as_eigen( vcam2.get_calibration().get_matrix() );
  EXPECT_MATRIX_EQ( calibration, calibration2 );

  auto const calibration_err =
    ( vcam.get_calibration().get_matrix() -
      vcam2.get_calibration().get_matrix() ).absolute_value_max();
  EXPECT_NEAR( 0.0, calibration_err, eps );

  std::cerr << "rotation: " << mcam.rotation().quaternion().coeffs() << std::endl;

  auto const& rotation = as_eigen( vcam.get_rotation().as_quaternion() );
  auto const& rotation2 = as_eigen( vcam2.get_rotation().as_quaternion() );
  EXPECT_MATRIX_EQ( rotation.coeffs(), rotation2.coeffs() );

  auto const rotation_err =
    ( vcam.get_rotation().as_quaternion() -
      vcam2.get_rotation().as_quaternion() ).inf_norm();
  EXPECT_NEAR( 0.0, rotation_err, eps );

  auto const& center = as_eigen( vcam.get_camera_center() );
  auto const& center2 = as_eigen( vcam2.get_camera_center() );
  EXPECT_MATRIX_EQ( center, center2 );

  auto const center_err =
    ( vcam.get_camera_center() -
      vcam2.get_camera_center() ).length();
  EXPECT_NEAR( 0.0, center_err, eps );
}

// ----------------------------------------------------------------------------
TEST(camera, convert_camera_double)
{
  test_convert_camera<double>(1e-15);
}

// ----------------------------------------------------------------------------
TEST(camera, convert_camera_float)
{
  test_convert_camera<float>(1e-7f);
}
