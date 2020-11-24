// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test dynamic configuration
 */

#include <arrows/uuid/uuid_factory_uuid.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::uuid;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ------------------------------------------------------------------
TEST(uuid, test_api)
{
  kac::uuid_factory_uuid algo;

  auto cfg = kwiver::vital::config_block::empty_config();

  EXPECT_TRUE( algo.check_configuration( cfg ) );

  kwiver::vital::uid id = algo.create_uuid();
  EXPECT_TRUE( id.is_valid() );
}

// ------------------------------------------------------------------
TEST(uuid, test_loading)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  auto cfg = kwiver::vital::config_block::empty_config();

  cfg->set_value( "uuid_cfg:type", "uuid" );

  algo::uuid_factory_sptr fact;

  // Check config so it will give run-time diagnostic if any config problems are found
  ASSERT_TRUE(
    algo::uuid_factory::check_nested_algo_configuration( "uuid_cfg", cfg ) );

  // Instantiate the configured algorithm
  algo::uuid_factory::set_nested_algo_configuration( "uuid_cfg", cfg, fact );
  ASSERT_NE( nullptr, fact ) << "Unable to create algorithm";
  EXPECT_EQ( "uuid", fact->impl_name() );
}
