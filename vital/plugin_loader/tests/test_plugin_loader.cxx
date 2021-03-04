// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(plugin_loader, module_marking)
{
  const auto module = plugin_manager::module_t{ "module" };
  plugin_manager& vpm = plugin_manager::instance();

  EXPECT_FALSE( vpm.is_module_loaded( module ) );

  vpm.mark_module_as_loaded( module );

  EXPECT_TRUE( vpm.is_module_loaded( module ) );
}

// Tests to add
//
// - Load known file and test to see if contents are as expected.
// - Test API

// - test reload by loading a set of plugins, add one more plugin,
// - test for that plugin(present), reload plugins, test for plugin(not there)
//
