/*ckwg +29
 * Copyright 2012-2018 by Kitware, Inc.
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

#include <test_common.h>

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>
#include <sprokit/pipeline_util/load_pipe_exception.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>

#include <kwiversys/SystemTools.hxx>

#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <cstdlib>

#define TEST_ARGS (kwiver::vital::path_t const& pipe_file)

DECLARE_TEST_MAP();

/// \todo Add tests for clusters without ports or processes.

static std::string const pipe_ext = ".pipe";

int
main( int argc, char* argv[] )
{
  CHECK_ARGS( 2 );

  std::string const testname = argv[1];
  kwiver::vital::path_t const pipe_dir = argv[2];

  kwiver::vital::path_t const pipe_file = pipe_dir + "/"  + testname + pipe_ext;

  RUN_TEST( testname, pipe_file );
}


// ----------------------------------------------------------------------------
sprokit::pipe_blocks load_pipe_blocks_from_file( kwiver::vital::path_t const& pipe_file )
{
  sprokit::pipeline_builder builder;
  builder.load_pipeline( pipe_file );
  return builder.pipeline_blocks();
}


// ----------------------------------------------------------------------------
sprokit::cluster_blocks load_cluster_blocks_from_file( kwiver::vital::path_t const& pipe_file )
{
  sprokit::pipeline_builder builder;
  builder.load_pipeline( pipe_file );
  return builder.cluster_blocks();
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_block )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_block_block )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  {
    const auto mykey = kwiver::vital::config_block_key_t( "myblock:foo:mykey" );
    const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
    const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

    if ( myvalue != expected )
    {
      TEST_ERROR( "Configuration was not correct: Expected: "
                  << expected << " Received: " << myvalue );
    }
  }

  {
    const auto mykey = kwiver::vital::config_block_key_t( "myblock:foo:oldkey" );
    const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
    const auto expected = kwiver::vital::config_block_value_t( "oldvalue" );

    if ( myvalue != expected )
    {
      TEST_ERROR( "Configuration was not correct: Expected: "
                  << expected << " Received: " << myvalue );
    }
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_block_relativepath )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:otherkey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const std::string cwd = kwiversys::SystemTools::GetFilenamePath( pipe_file );
  const auto expected = cwd + "/" + "value";

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_block_long_block )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:foo:a:b:c:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_block_nested_block )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:foo:bar:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_block_notalnum )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "my_block:my-key" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << expected << " Received: " << myvalue );
  }

  const auto myotherkey = kwiver::vital::config_block_key_t( "my-block:my_key" );
  const auto myothervalue = conf->get_value< kwiver::vital::config_block_value_t > ( myotherkey );
  const auto otherexpected = kwiver::vital::config_block_value_t( "myothervalue" );

  if ( myothervalue != otherexpected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << otherexpected << " Received: " << myothervalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_value_spaces )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "my value with spaces" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << expected << " Received: " << myvalue );
  }

  const auto mytabkey = kwiver::vital::config_block_key_t( "myblock:mytabs" );
  const auto mytabvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mytabkey );
  const auto tabexpected = kwiver::vital::config_block_value_t( "my	value	with	tabs");

  if ( mytabvalue != tabexpected )
  {
    TEST_ERROR( "Configuration was not correct: Expected: "
                << tabexpected << " Received: " << mytabvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_overrides )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myothervalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not overridden: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_read_only )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const config = sprokit::extract_configuration( blocks );

  const auto rokey = kwiver::vital::config_block_key_t( "myblock:mykey" );

  if ( ! config->is_read_only( rokey ) )
  {
    TEST_ERROR( "The configuration value was not marked as read only" );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_not_a_flag )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  EXPECT_EXCEPTION( sprokit::unrecognized_config_flag_exception,
                    sprokit::extract_configuration( blocks ),
                    "using an unknown flag" );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_read_only_override )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  EXPECT_EXCEPTION( kwiver::vital::set_on_read_only_value_exception,
                    sprokit::extract_configuration( blocks ),
                    "setting a read-only value" );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_append_ro )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration value was not appended: Expected: "
                << expected << " Received: " << myvalue );
  }

  if ( ! conf->is_read_only( mykey ) )
  {
    TEST_ERROR( "The configuration value was not marked as read only" );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_append_provided )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration value was not appended: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_append_provided_ro )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration value was not appended: Expected: "
                << expected << " Received: " << myvalue );
  }

  if ( ! conf->is_read_only( mykey ) )
  {
    TEST_ERROR( "The configuration value was not marked as read only" );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_append_comma )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue,othervalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration value was not appended with a comma separator: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_append_space_empty )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "othervalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration value was created with a space separator: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_append_path )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  kwiver::vital::path_t const expected = kwiver::vital::path_t( "myvalue" ) + "/" + kwiver::vital::path_t( "othervalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration value was not appended with a path separator: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_dotted_key )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:dotted.key" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "value" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not read properly: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_dotted_nested_key )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:dotted:nested.key:subkey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "value" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not read properly: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_provider_conf )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const conf = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myotherblock:mykey" );
  const auto myvalue = conf->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "myvalue" );

  if ( myvalue != expected )
  {
    TEST_ERROR( "Configuration was not overridden: Expected: "
                << expected << " Received: " << myvalue );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_provider_conf_dep )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  EXPECT_EXCEPTION( sprokit::provider_error_exception,
                    sprokit::extract_configuration( blocks ),
                    "Referencing config key not defined yet" );
}


// ------------------------------------------------------------------
TEST_PROPERTY( ENVIRONMENT, TEST_ENV = expected )
IMPLEMENT_TEST( config_provider_env )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const config = sprokit::extract_configuration( blocks );

  const auto mykey = kwiver::vital::config_block_key_t( "myblock:myenv" );
  const auto value = config->get_value< kwiver::vital::config_block_value_t > ( mykey );
  const auto expected = kwiver::vital::config_block_value_t( "expected" );

  if ( value != expected )
  {
    TEST_ERROR( "Environment was not read properly: Expected: "
                << expected << " Received: " << value );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_provider_read_only )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  kwiver::vital::config_block_sptr const config = sprokit::extract_configuration( blocks );

  const auto rokey = kwiver::vital::config_block_key_t( "myblock:mykey" );

  if ( ! config->is_read_only( rokey ) )
  {
    TEST_ERROR( "The configuration value was not marked as read only" );
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( config_provider_read_only_override )
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file( pipe_file );

  EXPECT_EXCEPTION( kwiver::vital::set_on_read_only_value_exception,
                    sprokit::extract_configuration( blocks ),
                    "setting a read-only provided value" );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( pipeline_multiplier )
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();
  sprokit::pipeline_builder builder;
  builder.load_pipeline( pipe_file );
  sprokit::pipeline_t const pipeline = builder.pipeline();

  if ( ! pipeline )
  {
    TEST_ERROR( "A pipeline was not created" );

    return;
  }

  pipeline->process_by_name( "gen_numbers1" );
  pipeline->process_by_name( "gen_numbers2" );
  pipeline->process_by_name( "multiply" );
  pipeline->process_by_name( "print" );

  /// \todo Verify the connections are done properly.
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_multiplier )
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::pipeline_builder builder;
  builder.load_cluster( pipe_file );
  sprokit::cluster_info_t const info = builder.cluster_info();
  const auto ctor = info->ctor;
  const auto config = kwiver::vital::config_block::empty_config();

  std::stringstream str;
  str << int(30);
  config->set_value( "factor", str.str() );

  sprokit::process_t const proc = ctor( config );

  sprokit::process_cluster_t const cluster = std::dynamic_pointer_cast< sprokit::process_cluster > ( proc );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  /// \todo Verify the configuration.
  /// \todo Verify the input mapping.
  /// \todo Verify the output mapping.
  /// \todo Verify the connections are done properly.
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_missing_cluster )
{
  (void)pipe_file;

  sprokit::cluster_blocks blocks;

  sprokit::process_pipe_block const pipe_block = sprokit::process_pipe_block();
  sprokit::cluster_block const block = pipe_block;

  blocks.push_back( block );

  EXPECT_EXCEPTION( sprokit::missing_cluster_block_exception,
                    sprokit::bake_cluster_blocks( blocks ),
                    "baking a set of cluster blocks without a cluster" );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_missing_processes )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  EXPECT_EXCEPTION( sprokit::cluster_without_processes_exception,
                    sprokit::bake_cluster_blocks( blocks ),
                    "baking a cluster without processes" );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_missing_ports )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  EXPECT_EXCEPTION( sprokit::cluster_without_ports_exception,
                    sprokit::bake_cluster_blocks( blocks ),
                    "baking a cluster without ports" );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_multiple_cluster )
{
  (void)pipe_file;

  sprokit::cluster_blocks blocks;

  sprokit::cluster_pipe_block const cluster_pipe_block = sprokit::cluster_pipe_block();
  sprokit::cluster_block const cluster_block = cluster_pipe_block;

  blocks.push_back( cluster_block );
  blocks.push_back( cluster_block );

  EXPECT_EXCEPTION( sprokit::multiple_cluster_blocks_exception,
                    sprokit::bake_cluster_blocks( blocks ),
                    "baking a set of cluster blocks without multiple clusters" );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_duplicate_input )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  EXPECT_EXCEPTION( sprokit::duplicate_cluster_input_port_exception,
                    sprokit::bake_cluster_blocks( blocks ),
                    "baking a cluster with duplicate input ports" );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_duplicate_output )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  EXPECT_EXCEPTION( sprokit::duplicate_cluster_output_port_exception,
                    sprokit::bake_cluster_blocks( blocks ),
                    "baking a cluster with duplicate output ports" );
}

static void test_cluster( sprokit::process_t const& cluster, std::string const& path );

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_configuration_default )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks( blocks );
  const auto ctor = info->ctor;
  const auto config = kwiver::vital::config_block::empty_config();

  sprokit::process_t const cluster = ctor( config );

  std::string const output_path = "test-pipe_bakery-configuration_default.txt";

  test_cluster( cluster, output_path );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_configuration_provide )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks( blocks );
  const auto ctor = info->ctor;
  const auto config = kwiver::vital::config_block::empty_config();

  sprokit::process_t const cluster = ctor( config );

  std::string const output_path = "test-pipe_bakery-configuration_provide.txt";

  test_cluster( cluster, output_path );
}

static sprokit::process_cluster_t setup_map_config_cluster( sprokit::process::name_t const& name, kwiver::vital::path_t const& pipe_file );

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_map_config )
{
  /// \todo This test is poorly designed in that it tests for mapping by
  /// leveraging the fact that only mapped configuration get pushed through when
  /// reconfiguring a cluster.

  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "expected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  pipeline->reconfigure( new_conf );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_map_config_tunable )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "expected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  pipeline->reconfigure( new_conf );
}

#if 0  // disable incomplete tests
//+ Need to mangle the macro name so the CMake tooling does not
//+  register these tests even though they are not active
//------------------------------------------------------------------
DONT_IMPLEMENT_TEST( cluster_map_config_redirect )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "expected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  sprokit::process::name_t const proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep +
                       proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}


// ------------------------------------------------------------------
DONT_IMPLEMENT_TEST( cluster_map_config_modified )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "expected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  sprokit::process::name_t const proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}
#endif


// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_map_config_not_read_only )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "unexpected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  sprokit::process::name_t const proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_map_config_only_provided )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "unexpected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  sprokit::process::name_t const proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}

// ------------------------------------------------------------------
TEST_PROPERTY( ENVIRONMENT, TEST_ENV = expected )
IMPLEMENT_TEST( cluster_map_config_only_conf_provided )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "unexpected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  sprokit::process::name_t const proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_map_config_to_non_process )
{
  /// \todo This test is poorly designed because there's no check that / the
  /// mapping wasn't actually created.

  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto key = kwiver::vital::config_block_key_t( "tunable" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "expected" );

  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + key, tuned_value );

  sprokit::process::name_t const proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_map_config_not_from_cluster )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  sprokit::process_cluster_t const cluster = setup_map_config_cluster( cluster_name, pipe_file );

  if ( ! cluster )
  {
    TEST_ERROR( "A cluster was not created" );

    return;
  }

  sprokit::pipeline_t const pipeline = std::make_shared< sprokit::pipeline > ( kwiver::vital::config_block::empty_config() );

  pipeline->add_process( cluster );
  pipeline->setup_pipeline();

  const auto new_conf = kwiver::vital::config_block::empty_config();
  const auto proc_name = sprokit::process::name_t( "expect" );
  const auto new_key = kwiver::vital::config_block_key_t( "new_key" );
  const auto tuned_value = kwiver::vital::config_block_key_t( "expected" );

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value( cluster_name + kwiver::vital::config_block::block_sep + proc_name + kwiver::vital::config_block::block_sep + new_key,
                       tuned_value );

  pipeline->reconfigure( new_conf );
}

// ------------------------------------------------------------------
IMPLEMENT_TEST( cluster_override_mapped )
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t( "cluster" );

  const auto conf = kwiver::vital::config_block::empty_config();
  const auto proc_name = sprokit::process::name_t( "expect" );
  const auto key = kwiver::vital::config_block_key_t( "tunable" );

  const auto full_key = proc_name + kwiver::vital::config_block::block_sep + key;
  const auto value = kwiver::vital::config_block_value_t( "unexpected" );

  // The problem here is that the expect:tunable parameter is meant to be mapped
  // with the map_config call within the cluster. Due to the implementation,
  // the lines which cause the linking are removed, so if we try to sneak in an
  // invalid parameter, we should get an error about overriding a read-only
  // variable.
  conf->set_value( full_key, value );

  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks( blocks );
  const auto ctor = info->ctor;

  EXPECT_EXCEPTION( kwiver::vital::set_on_read_only_value_exception,
                    ctor( conf ),
                    "manually setting a parameter which is mapped in a cluster" );
}

static sprokit::process_t create_process( sprokit::process::type_t const& type,
                                          sprokit::process::name_t const& name,
                                          kwiver::vital::config_block_sptr config = kwiver::vital::config_block::empty_config() );
static sprokit::pipeline_t create_pipeline();

// ------------------------------------------------------------------
void
test_cluster( sprokit::process_t const& cluster, std::string const& path )
{
  sprokit::process::type_t const proc_typeu = sprokit::process::type_t( "numbers" );
  sprokit::process::type_t const proc_typet = sprokit::process::type_t( "print_number" );

  sprokit::process::name_t const proc_nameu = sprokit::process::name_t( "upstream" );
  sprokit::process::name_t const proc_named = cluster->name();
  sprokit::process::name_t const proc_namet = sprokit::process::name_t( "terminal" );

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  {
    const auto configu = kwiver::vital::config_block::empty_config();

    const auto start_key = kwiver::vital::config_block_key_t( "start" );
    const auto end_key = kwiver::vital::config_block_key_t( "end" );

    std::stringstream str;
    str << start_value;
    const auto start_num = str.str();

    str.clear();
    str << end_value;
    const auto end_num = str.str();

    configu->set_value( start_key, start_num );
    configu->set_value( end_key, end_num );

    const auto configt = kwiver::vital::config_block::empty_config();

    const auto output_key = kwiver::vital::config_block_key_t( "output" );
    const auto output_value = kwiver::vital::config_block_value_t( path );

    configt->set_value( output_key, output_value );

    sprokit::process_t const processu = create_process( proc_typeu, proc_nameu, configu );
    sprokit::process_t const processt = create_process( proc_typet, proc_namet, configt );

    sprokit::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process( processu );
    pipeline->add_process( cluster );
    pipeline->add_process( processt );

    const auto port_nameu = sprokit::process::port_t( "number" );
    const auto port_namedi = sprokit::process::port_t( "factor" );
    const auto port_namedo = sprokit::process::port_t( "product" );
    const auto port_namet = sprokit::process::port_t( "number" );

    pipeline->connect( proc_nameu, port_nameu,
                       proc_named, port_namedi );
    pipeline->connect( proc_named, port_namedo,
                       proc_namet, port_namet );

    pipeline->setup_pipeline();

    sprokit::scheduler_t const scheduler = sprokit::create_scheduler( sprokit::scheduler_factory::default_type, pipeline );

    scheduler->start();
    scheduler->wait();
  }

  std::ifstream fin( path.c_str() );

  if ( ! fin.good() )
  {
    TEST_ERROR( "Could not open the output file" );
  }

  std::string line;

  // From the input cluster.
  static const int32_t factor = 20;

  for ( int32_t i = start_value; i < end_value; ++i )
  {
    if ( ! std::getline( fin, line ) )
    {
      TEST_ERROR( "Failed to read a line from the file" );
    }

    std::stringstream str;
    str << ( i * factor );
    if ( kwiver::vital::config_block_value_t( line ) != str.str() )
    {
      TEST_ERROR( "Did not get expected value: Expected: "
                  << (i * factor) << " Received: " << line );
    }
  }

  if ( std::getline( fin, line ) )
  {
    TEST_ERROR( "More results than expected in the file" );
  }

  if ( ! fin.eof() )
  {
    TEST_ERROR( "Not at end of file" );
  }
} // test_cluster


sprokit::process_t
create_process( sprokit::process::type_t const& type, sprokit::process::name_t const& name, kwiver::vital::config_block_sptr config )
{
  static bool const modules_loaded = ( kwiver::vital::plugin_manager::instance().load_all_plugins(), true );

  (void)modules_loaded;

  return sprokit::create_process( type, name, config );
}


sprokit::pipeline_t
create_pipeline()
{
  return std::make_shared< sprokit::pipeline > ();
}


sprokit::process_cluster_t
setup_map_config_cluster( sprokit::process::name_t const& name, kwiver::vital::path_t const& pipe_file )
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file( pipe_file );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks( blocks );
  const auto ctor = info->ctor;
  const auto config = kwiver::vital::config_block::empty_config();

  config->set_value( sprokit::process::config_name, name );

  sprokit::process_t const proc = ctor( config );

  sprokit::process_cluster_t const cluster = std::dynamic_pointer_cast< sprokit::process_cluster > ( proc );

  return cluster;
}
