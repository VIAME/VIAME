/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/pipe_bakery.h>
#include <vistk/pipeline_util/pipe_bakery_exception.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process_cluster.h>

#include <boost/lexical_cast.hpp>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

#include <cstdlib>

static std::string const pipe_ext = ".pipe";

static void run_test(std::string const& test_name, vistk::path_t const& pipe_file);

int
main(int argc, char* argv[])
{
  if (argc != 3)
  {
    TEST_ERROR("Expected two arguments");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];
  vistk::path_t const pipe_dir = argv[2];

  vistk::path_t const pipe_file = pipe_dir / (test_name + pipe_ext);

  try
  {
    run_test(test_name, pipe_file);
  }
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_config_block(vistk::path_t const& pipe_file);
static void test_config_block_notalnum(vistk::path_t const& pipe_file);
static void test_config_value_spaces(vistk::path_t const& pipe_file);
static void test_config_overrides(vistk::path_t const& pipe_file);
static void test_config_read_only(vistk::path_t const& pipe_file);
static void test_config_not_a_flag(vistk::path_t const& pipe_file);
static void test_config_read_only_override(vistk::path_t const& pipe_file);
static void test_config_append(vistk::path_t const& pipe_file);
static void test_config_cappend(vistk::path_t const& pipe_file);
static void test_config_cappend_empty(vistk::path_t const& pipe_file);
static void test_config_provider_conf(vistk::path_t const& pipe_file);
static void test_config_provider_conf_dep(vistk::path_t const& pipe_file);
static void test_config_provider_conf_circular_dep(vistk::path_t const& pipe_file);
static void test_config_provider_env(vistk::path_t const& pipe_file);
static void test_config_provider_read_only(vistk::path_t const& pipe_file);
static void test_config_provider_read_only_override(vistk::path_t const& pipe_file);
static void test_config_provider_unprovided(vistk::path_t const& pipe_file);
static void test_pipeline_multiplier(vistk::path_t const& pipe_file);
static void test_cluster_multiplier(vistk::path_t const& pipe_file);
static void test_cluster_missing_cluster(vistk::path_t const& pipe_file);
static void test_cluster_multiple_cluster(vistk::path_t const& pipe_file);
static void test_cluster_duplicate_input(vistk::path_t const& pipe_file);
static void test_cluster_duplicate_output(vistk::path_t const& pipe_file);

/// \todo Add tests for clusters without ports or processes.

void
run_test(std::string const& test_name, vistk::path_t const& pipe_file)
{
  if (test_name == "config_block")
  {
    test_config_block(pipe_file);
  }
  else if (test_name == "config_block_notalnum")
  {
    test_config_block_notalnum(pipe_file);
  }
  else if (test_name == "config_value_spaces")
  {
    test_config_value_spaces(pipe_file);
  }
  else if (test_name == "config_overrides")
  {
    test_config_overrides(pipe_file);
  }
  else if (test_name == "config_read_only")
  {
    test_config_read_only(pipe_file);
  }
  else if (test_name == "config_not_a_flag")
  {
    test_config_not_a_flag(pipe_file);
  }
  else if (test_name == "config_read_only_override")
  {
    test_config_read_only_override(pipe_file);
  }
  else if (test_name == "config_append")
  {
    test_config_append(pipe_file);
  }
  else if (test_name == "config_cappend")
  {
    test_config_cappend(pipe_file);
  }
  else if (test_name == "config_cappend_empty")
  {
    test_config_cappend_empty(pipe_file);
  }
  else if (test_name == "config_provider_conf")
  {
    test_config_provider_conf(pipe_file);
  }
  else if (test_name == "config_provider_conf_dep")
  {
    test_config_provider_conf_dep(pipe_file);
  }
  else if (test_name == "config_provider_conf_circular_dep")
  {
    test_config_provider_conf_circular_dep(pipe_file);
  }
  else if (test_name == "config_provider_env")
  {
    test_config_provider_env(pipe_file);
  }
  else if (test_name == "config_provider_read_only")
  {
    test_config_provider_read_only(pipe_file);
  }
  else if (test_name == "config_provider_read_only_override")
  {
    test_config_provider_read_only_override(pipe_file);
  }
  else if (test_name == "config_provider_unprovided")
  {
    test_config_provider_unprovided(pipe_file);
  }
  else if (test_name == "pipeline_multiplier")
  {
    test_pipeline_multiplier(pipe_file);
  }
  else if (test_name == "cluster_multiplier")
  {
    test_cluster_multiplier(pipe_file);
  }
  else if (test_name == "cluster_missing_cluster")
  {
    test_cluster_missing_cluster(pipe_file);
  }
  else if (test_name == "cluster_multiple_cluster")
  {
    test_cluster_multiple_cluster(pipe_file);
  }
  else if (test_name == "cluster_duplicate_input")
  {
    test_cluster_duplicate_input(pipe_file);
  }
  else if (test_name == "cluster_duplicate_output")
  {
    test_cluster_duplicate_output(pipe_file);
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_config_block(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

void
test_config_block_notalnum(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("my_block:my-key");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  vistk::config::key_t const myotherkey = vistk::config::key_t("my-block:my_key");
  vistk::config::value_t const myothervalue = conf->get_value<vistk::config::value_t>(myotherkey);
  vistk::config::value_t const otherexpected = vistk::config::value_t("myothervalue");

  if (myothervalue != otherexpected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << otherexpected << " "
               "Received: " << myothervalue);
  }
}

void
test_config_value_spaces(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("my value with spaces");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  vistk::config::key_t const mytabkey = vistk::config::key_t("myblock:mytabs");
  vistk::config::value_t const mytabvalue = conf->get_value<vistk::config::value_t>(mytabkey);
  vistk::config::value_t const tabexpected = vistk::config::value_t("my	value	with	tabs");

  if (mytabvalue != tabexpected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << tabexpected << " "
               "Received: " << mytabvalue);
  }
}

void
test_config_overrides(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myothervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

void
test_config_read_only(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const config = vistk::extract_configuration(blocks);

  vistk::config::key_t const rokey = vistk::config::key_t("myblock:mykey");

  if (!config->is_read_only(rokey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

void
test_config_not_a_flag(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::unrecognized_config_flag_exception,
                   vistk::extract_configuration(blocks),
                   "using an unknown flag");
}

void
test_config_read_only_override(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   vistk::extract_configuration(blocks),
                   "setting a read-only value");
}

void
test_config_append(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

void
test_config_cappend(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myvalue,othervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended with a comma separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

void
test_config_cappend_empty(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("othervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was created with a comma separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

void
test_config_provider_conf(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myotherblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

void
test_config_provider_conf_dep(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myotherblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  vistk::config::key_t const mymidkey = vistk::config::key_t("mymidblock:mykey");
  vistk::config::value_t const mymidvalue = conf->get_value<vistk::config::value_t>(mymidkey);

  if (mymidvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << mymidvalue);
  }
}

void
test_config_provider_conf_circular_dep(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::circular_config_provide_exception,
                   vistk::extract_configuration(blocks),
                   "circular configuration provides exist");
}

void
test_config_provider_env(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const config = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:myenv");
  vistk::config::value_t const value = config->get_value<vistk::config::value_t>(mykey);
  vistk::config::value_t const expected = vistk::config::value_t("expected");

  if (value != expected)
  {
    TEST_ERROR("Environment was not read properly: "
               "Expected: " << expected << " "
               "Received: " << value);
  }
}

void
test_config_provider_read_only(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const config = vistk::extract_configuration(blocks);

  vistk::config::key_t const rokey = vistk::config::key_t("myblock:mykey");

  if (!config->is_read_only(rokey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

void
test_config_provider_read_only_override(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   vistk::extract_configuration(blocks),
                   "setting a read-only provided value");
}

void
test_config_provider_unprovided(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::unrecognized_provider_exception,
                   vistk::extract_configuration(blocks),
                   "using an unknown provider");
}

void
test_pipeline_multiplier(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  vistk::pipeline_t const pipeline = vistk::bake_pipe_blocks(blocks);

  if (!pipeline)
  {
    TEST_ERROR("A pipeline was not created");

    return;
  }

  pipeline->process_by_name("gen_numbers1");
  pipeline->process_by_name("gen_numbers2");
  pipeline->process_by_name("multiply");
  pipeline->process_by_name("print");

  /// \todo Verify the connections are done properly.
}

void
test_cluster_multiplier(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  vistk::cluster_info_t const info = vistk::bake_cluster_blocks(blocks);

  vistk::process_ctor_t const ctor = info->ctor;

  vistk::config_t const config = vistk::config::empty_config();

  config->set_value("factor", boost::lexical_cast<vistk::config::value_t>(30));

  vistk::process_t const proc = ctor(config);

  vistk::process_cluster_t const cluster = boost::dynamic_pointer_cast<vistk::process_cluster>(proc);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  /// \todo Verify the configuration.
  /// \todo Verify the input mapping.
  /// \todo Verify the output mapping.
  /// \todo Verify the connections are done properly.
}

void
test_cluster_missing_cluster(vistk::path_t const& /*pipe_file*/)
{
  vistk::cluster_blocks blocks;

  vistk::process_pipe_block const pipe_block = vistk::process_pipe_block();
  vistk::cluster_block const block = pipe_block;

  blocks.push_back(block);

  EXPECT_EXCEPTION(vistk::missing_cluster_block_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a set of cluster blocks without a cluster");
}

void
test_cluster_multiple_cluster(vistk::path_t const& /*pipe_file*/)
{
  vistk::cluster_blocks blocks;

  vistk::cluster_pipe_block const cluster_pipe_block = vistk::cluster_pipe_block();
  vistk::cluster_block const cluster_block = cluster_pipe_block;

  blocks.push_back(cluster_block);
  blocks.push_back(cluster_block);

  EXPECT_EXCEPTION(vistk::multiple_cluster_blocks_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a set of cluster blocks without multiple clusters");
}

void
test_cluster_duplicate_input(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  EXPECT_EXCEPTION(vistk::duplicate_cluster_input_port_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a cluster with duplicate input ports");
}

void
test_cluster_duplicate_output(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  EXPECT_EXCEPTION(vistk::duplicate_cluster_output_port_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a cluster with duplicate output ports");
}
