/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
 * \brief test core rotation class
 */

#include <test_eigen.h>

#include <vital/types/rotation.h>

#include <iostream>
#include <vector>


static constexpr double pi = 3.14159265358979323846;

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(rotation, default_constructor)
{
  rotation_d rot;
  EXPECT_EQ( Eigen::Quaterniond::Identity(), rot.quaternion() );
  EXPECT_EQ( Eigen::Matrix3d::Identity(), rot.matrix() );
}

// ----------------------------------------------------------------------------
TEST(rotation, convert_rodrigues)
{
  EXPECT_EQ( ( vector_3d{ 0, 0, 0 } ), rotation_d{}.rodrigues() );
  EXPECT_EQ( rotation_d{}, ( rotation_d{ vector_3d{ 0, 0, 0 } } ) );

  vector_3d rvec{ 2, -1, 0.5 };
  rotation_d rot{ rvec };

  std::cerr << "rvec magnitude: " << rvec.norm() << std::endl;
  std::cerr << "rot3 magnitude: " << rot.rodrigues().norm() << std::endl;

  EXPECT_MATRIX_NEAR( rvec, rot.rodrigues(), 1e-14 );
  EXPECT_MATRIX_NEAR( rvec.normalized(), rot.axis(), 1e-14 );
}

// ----------------------------------------------------------------------------
TEST(rotation, convert_axis_angle)
{
  double angle = 0.8;
  vector_3d axis = vector_3d{ -3, 2, 1 }.normalized();

  rotation_d rot{ angle, axis };

  EXPECT_NEAR( angle, rot.angle(), 1e-14 );
  EXPECT_MATRIX_NEAR( axis, rot.axis(), 1e-14 );
}

// ----------------------------------------------------------------------------
struct ypr_test
{
  double yaw;
  double pitch;
  double roll;
};

// ----------------------------------------------------------------------------
void
PrintTo( ypr_test const& v, ::std::ostream* os )
{
  (*os) << "Y:" << v.yaw << ", P:" << v.pitch << ", R:" << v.roll;
}

// ----------------------------------------------------------------------------
class rotation_yaw_pitch_roll : public ::testing::TestWithParam<ypr_test>
{
};

// ----------------------------------------------------------------------------
TEST_P(rotation_yaw_pitch_roll, convert)
{
  auto const& p = GetParam();
  rotation_d rot{ p.yaw, p.pitch, p.roll };

  std::cout << "rotation matrix:" << std::endl << rot.matrix() << std::endl;

  double extracted_yaw, extracted_pitch, extracted_roll;
  rot.get_yaw_pitch_roll( extracted_yaw, extracted_pitch, extracted_roll );

  EXPECT_NEAR( p.yaw,   extracted_yaw,   1e-14 );
  EXPECT_NEAR( p.pitch, extracted_pitch, 1e-14 );
  EXPECT_NEAR( p.roll,  extracted_roll,  1e-14 );
}

// ----------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(
  ,
  rotation_yaw_pitch_roll,
  ::testing::Values(
      ( ypr_test{  0.0,  0.0,  0.0 } ),
      ( ypr_test{ +1.2,  0.0,  0.0 } ),
      ( ypr_test{  0.0, +0.3,  0.0 } ),
      ( ypr_test{  0.0,  0.0, -1.7 } ),
      ( ypr_test{  0.0, +0.3, -1.7 } ),
      ( ypr_test{ +1.2,  0.0, -1.7 } ),
      ( ypr_test{ +1.2, +0.3,  0.0 } ),
      ( ypr_test{ +1.2, +0.3, -1.7 } )
  ) );

// ----------------------------------------------------------------------------
TEST(rotation, compose)
{
  rotation_d rot1{ vector_3d{  0.1, -1.5, 2.0 } };
  rotation_d rot2{ vector_3d{ -0.5, -0.5, 1.0 } };

  EXPECT_MATRIX_NEAR( ( rot1.matrix() * rot2.matrix() ).eval(),
                      ( rot1 * rot2 ).matrix(), 1e-14 );
}

// ----------------------------------------------------------------------------
TEST(rotation, interpolation)
{
  rotation_d x{ 0, vector_3d{ 1, 0, 0 } };
  rotation_d y{ pi / 2, vector_3d{ 0, 1, 0 } };
  rotation_d z = interpolate_rotation( x, y, 0.5 );

  std::cerr << "x: " << x.axis() << " " << x.angle() << std::endl
            << "y: " << y.axis() << " " << y.angle() << std::endl
            << "z: " << z.axis() << " " << z.angle() << std::endl;

  EXPECT_MATRIX_NEAR( ( vector_3d{ 0, 1, 0 } ), z.axis(), 1e-15 );
  EXPECT_NEAR( pi / 4, z.angle(), 1e-15 );
}

// ----------------------------------------------------------------------------
TEST(rotation, multiple_interpolations)
{
  rotation_d x{ 0, vector_3d{ 1, 0, 0 } };
  rotation_d y{ pi / 2, vector_3d{ 0, 1, 0 } };
  std::vector<rotation_d> rots;

  rots.push_back( x );
  interpolated_rotations( x, y, 3, rots );
  rots.push_back( y );

  ASSERT_EQ( 5, rots.size() );
  for ( rotation_d rot : rots )
  {
    std::cerr << "\t" << rot.axis() << ' ' << rot.angle() << std::endl;
  }

  auto i1 = rots[1];
  auto i2 = rots[2];
  auto i3 = rots[3];

  std::cerr << "i1 .25 : " << i1.axis() << ' ' << i1.angle() << std::endl;
  std::cerr << "i2 .50 : " << i2.axis() << ' ' << i2.angle() << std::endl;
  std::cerr << "i3 .75 : " << i3.axis() << ' ' << i3.angle() << std::endl;

  EXPECT_MATRIX_NEAR( ( vector_3d{ 0, 1, 0 } ), i1.axis(), 1e-15 );
  EXPECT_NEAR( ( 1 * pi ) / 8, i1.angle(), 1e-15 );

  EXPECT_MATRIX_NEAR( ( vector_3d{ 0, 1, 0 } ), i2.axis(), 1e-15 );
  EXPECT_NEAR( ( 2 * pi ) / 8, i2.angle(), 1e-15 );

  EXPECT_MATRIX_NEAR( ( vector_3d{ 0, 1, 0 } ), i3.axis(), 1e-15 );
  EXPECT_NEAR( ( 3 * pi ) / 8, i3.angle(), 1e-15 );
}
