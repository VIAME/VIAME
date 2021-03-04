// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
