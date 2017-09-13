/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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
 * \brief core config_block tests
 */

#include <test_eigen.h>

#include <vital/config/config_block.h>
#include <vital/io/eigen_io.h>
#include <vital/types/vector.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(config, value_conversion)
{
  config_block_sptr const config = config_block::empty_config();
  config_block_key_t const key = config_block_key_t{ "key" };

  constexpr static double double_value = 123.456;
  config->set_value( key, double_value );
  EXPECT_DOUBLE_EQ( double_value, config->get_value<double>( key ) );

  constexpr static unsigned int uint_value = 1234567;
  config->set_value( key, uint_value );
  EXPECT_EQ( uint_value, config->get_value<unsigned int>( key ) );

  vector_2d const vec2d_value{ 2.34, 0.0567 };
  config->set_value( key, vec2d_value );
  EXPECT_MATRIX_EQ( vec2d_value, config->get_value<vector_2d>( key ) );

  std::string const string_value{ "some string" };
  config->set_value( key, string_value );
  EXPECT_EQ( string_value, config->get_value<std::string>( key ) );

  config_block_value_t cbv_value{ "some value string" };
  config->set_value( key, cbv_value );
  EXPECT_EQ( cbv_value, config->get_value<config_block_value_t>( key ) );
}
