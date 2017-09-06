/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
