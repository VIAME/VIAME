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
