/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
