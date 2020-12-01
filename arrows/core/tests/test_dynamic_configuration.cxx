// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test dynamic configuration
 */

#include <arrows/core/dynamic_config_none.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::core;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(dynamic_configuration, test_api)
{
  kac::dynamic_config_none dcn;

  auto cfg = kwiver::vital::config_block::empty_config();

  EXPECT_TRUE( dcn.check_configuration( cfg ) );

  cfg = dcn.get_dynamic_configuration();
  const auto values = cfg->available_values();
  EXPECT_EQ( 0, values.size() );
}

// ----------------------------------------------------------------------------
TEST(dynamic_configuration, test_loading)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  auto cfg = kwiver::vital::config_block::empty_config();

  cfg->set_value( "dyn_cfg:type", "none" );

  algo::dynamic_configuration_sptr dcs;

  // Check config so it will give run-time diagnostic if any config problems are found
  EXPECT_TRUE( algo::dynamic_configuration::check_nested_algo_configuration( "dyn_cfg", cfg ) );

  // Instantiate the configured algorithm
  algo::dynamic_configuration::set_nested_algo_configuration( "dyn_cfg", cfg, dcs );
  ASSERT_NE( nullptr, dcs ) << "Failed to create algorithm";
  EXPECT_EQ( "none", dcs->impl_name() );
}
