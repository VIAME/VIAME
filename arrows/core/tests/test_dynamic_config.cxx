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

#include <test_common.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <arrows/core/dynamic_config_none.h>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::core;

// ------------------------------------------------------------------
IMPLEMENT_TEST(test_api)
{
  kac::dynamic_config_none dcn;

  auto cfg = kwiver::vital::config_block::empty_config();

  TEST_EQUAL( "check_configuration return", dcn.check_configuration( cfg ), true );

  cfg = dcn.get_dynamic_configuration();
  const auto values = cfg->available_values();
  TEST_EQUAL( "empty config", values.size(), 0 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(test_loading)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  auto cfg = kwiver::vital::config_block::empty_config();

  cfg->set_value( "dyn_cfg:type", "none" );

  algo::dynamic_configuration_sptr dcs;

  // Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::dynamic_configuration::check_nested_algo_configuration( "dyn_cfg", cfg ) )
  {
    TEST_ERROR( "Configuration check failed." );
  }

  // Instantiate the configured algorithm
  algo::dynamic_configuration::set_nested_algo_configuration( "dyn_cfg", cfg, dcs );
  if ( ! dcs )
  {
    TEST_ERROR( "Unable to create algorithm" );
  }
  else
  {
    TEST_EQUAL( "algorithm name", dcs->impl_name(), "none" );
  }
}
