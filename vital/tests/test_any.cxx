// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test core any class
 */

#include <vital/any.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(any, api)
{
  kwiver::vital::any any_one;
  EXPECT_TRUE( any_one.empty() );

  kwiver::vital::any any_string( std::string ("this is a string") );
  EXPECT_FALSE( any_string.empty() );

  // Clear any and test it
  any_string.clear();
  EXPECT_TRUE( any_string.empty() );
  EXPECT_EQ( any_one.type(), any_string.type() );

  kwiver::vital::any any_double(3.14159);
  EXPECT_FLOAT_EQ( 3.14159, kwiver::vital::any_cast<double>( any_double ) );
  EXPECT_FLOAT_EQ( 3.14159, *kwiver::vital::any_cast<double>( &any_double ) );

  EXPECT_THROW(
    kwiver::vital::any_cast<std::string >( any_double ),
    kwiver::vital::bad_any_cast );

  kwiver::vital::any new_double = any_double;
  EXPECT_EQ( any_double.type(), new_double.type() );

  // Update through pointer
  double* dptr = kwiver::vital::any_cast<double>( &any_double );
  EXPECT_FLOAT_EQ( 3.14159, *dptr );
  *dptr = 6.28318;
  EXPECT_FLOAT_EQ( 6.28318, kwiver::vital::any_cast<double>( any_double ) );
  EXPECT_FLOAT_EQ( 6.28318, *dptr );

  // Update through assignment
  any_double = 3.14159;
  EXPECT_FLOAT_EQ( 3.14159, kwiver::vital::any_cast<double>( any_double ) );

  // Reset value and type of any
  any_double = 3456; // convert to int
  EXPECT_EQ( 3456, kwiver::vital::any_cast<int>( any_double ) );
}
