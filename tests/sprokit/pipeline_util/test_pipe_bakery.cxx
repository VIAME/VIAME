/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <sprokit/pipeline_util/load_pipe.h>
#include <sprokit/pipeline_util/path.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>

#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_registry.h>

#include <boost/lexical_cast.hpp>

#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cstdlib>

#define TEST_ARGS (sprokit::path_t const& pipe_file)

DECLARE_TEST(config_block);
DECLARE_TEST(config_block_notalnum);
DECLARE_TEST(config_value_spaces);
DECLARE_TEST(config_overrides);
DECLARE_TEST(config_read_only);
DECLARE_TEST(config_not_a_flag);
DECLARE_TEST(config_read_only_override);
DECLARE_TEST(config_append);
DECLARE_TEST(config_append_ro);
DECLARE_TEST(config_append_provided);
DECLARE_TEST(config_append_provided_ro);
DECLARE_TEST(config_append_comma);
DECLARE_TEST(config_append_comma_empty);
DECLARE_TEST(config_append_space);
DECLARE_TEST(config_append_space_empty);
DECLARE_TEST(config_append_path);
DECLARE_TEST(config_append_path_empty);
DECLARE_TEST(config_append_flag_mismatch_ac);
DECLARE_TEST(config_append_flag_mismatch_ap);
DECLARE_TEST(config_append_flag_mismatch_cp);
DECLARE_TEST(config_append_flag_mismatch_all);
DECLARE_TEST(config_provider_conf);
DECLARE_TEST(config_provider_conf_dep);
DECLARE_TEST(config_provider_conf_circular_dep);
DECLARE_TEST(config_provider_env);
DECLARE_TEST(config_provider_read_only);
DECLARE_TEST(config_provider_read_only_override);
DECLARE_TEST(config_provider_unprovided);
DECLARE_TEST(pipeline_multiplier);
DECLARE_TEST(cluster_multiplier);
DECLARE_TEST(cluster_missing_cluster);
DECLARE_TEST(cluster_missing_processes);
DECLARE_TEST(cluster_missing_ports);
DECLARE_TEST(cluster_multiple_cluster);
DECLARE_TEST(cluster_duplicate_input);
DECLARE_TEST(cluster_duplicate_output);
DECLARE_TEST(cluster_configuration_default);
DECLARE_TEST(cluster_configuration_provide);
DECLARE_TEST(cluster_map_config);
DECLARE_TEST(cluster_map_config_tunable);
DECLARE_TEST(cluster_map_config_redirect);
DECLARE_TEST(cluster_map_config_modified);
DECLARE_TEST(cluster_map_config_not_read_only);
DECLARE_TEST(cluster_map_config_only_provided);
DECLARE_TEST(cluster_map_config_only_conf_provided);
DECLARE_TEST(cluster_map_config_to_non_process);
DECLARE_TEST(cluster_map_config_not_from_cluster);
DECLARE_TEST(cluster_override_mapped);

/// \todo Add tests for clusters without ports or processes.

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  std::string const testname = argv[1];
  sprokit::path_t const pipe_dir = argv[2];

  sprokit::path_t const pipe_file = pipe_dir / (testname + pipe_ext);

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, config_block);
  ADD_TEST(tests, config_block_notalnum);
  ADD_TEST(tests, config_value_spaces);
  ADD_TEST(tests, config_overrides);
  ADD_TEST(tests, config_read_only);
  ADD_TEST(tests, config_not_a_flag);
  ADD_TEST(tests, config_read_only_override);
  ADD_TEST(tests, config_append);
  ADD_TEST(tests, config_append_ro);
  ADD_TEST(tests, config_append_provided);
  ADD_TEST(tests, config_append_provided_ro);
  ADD_TEST(tests, config_append_comma);
  ADD_TEST(tests, config_append_comma_empty);
  ADD_TEST(tests, config_append_space);
  ADD_TEST(tests, config_append_space_empty);
  ADD_TEST(tests, config_append_path);
  ADD_TEST(tests, config_append_path_empty);
  ADD_TEST(tests, config_append_flag_mismatch_ac);
  ADD_TEST(tests, config_append_flag_mismatch_ap);
  ADD_TEST(tests, config_append_flag_mismatch_cp);
  ADD_TEST(tests, config_append_flag_mismatch_all);
  ADD_TEST(tests, config_provider_conf);
  ADD_TEST(tests, config_provider_conf_dep);
  ADD_TEST(tests, config_provider_conf_circular_dep);
  ADD_TEST(tests, config_provider_env);
  ADD_TEST(tests, config_provider_read_only);
  ADD_TEST(tests, config_provider_read_only_override);
  ADD_TEST(tests, config_provider_unprovided);
  ADD_TEST(tests, pipeline_multiplier);
  ADD_TEST(tests, cluster_multiplier);
  ADD_TEST(tests, cluster_missing_cluster);
  ADD_TEST(tests, cluster_missing_processes);
  ADD_TEST(tests, cluster_missing_ports);
  ADD_TEST(tests, cluster_multiple_cluster);
  ADD_TEST(tests, cluster_duplicate_input);
  ADD_TEST(tests, cluster_duplicate_output);
  ADD_TEST(tests, cluster_configuration_default);
  ADD_TEST(tests, cluster_configuration_provide);
  ADD_TEST(tests, cluster_map_config);
  ADD_TEST(tests, cluster_map_config_tunable);
  ADD_TEST(tests, cluster_map_config_redirect);
  ADD_TEST(tests, cluster_map_config_modified);
  ADD_TEST(tests, cluster_map_config_not_read_only);
  ADD_TEST(tests, cluster_map_config_only_provided);
  ADD_TEST(tests, cluster_map_config_only_conf_provided);
  ADD_TEST(tests, cluster_map_config_to_non_process);
  ADD_TEST(tests, cluster_map_config_not_from_cluster);
  ADD_TEST(tests, cluster_override_mapped);

  RUN_TEST(tests, testname, pipe_file);
}

IMPLEMENT_TEST(config_block)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_block_notalnum)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("my_block:my-key");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  sprokit::config::key_t const myotherkey = sprokit::config::key_t("my-block:my_key");
  sprokit::config::value_t const myothervalue = conf->get_value<sprokit::config::value_t>(myotherkey);
  sprokit::config::value_t const otherexpected = sprokit::config::value_t("myothervalue");

  if (myothervalue != otherexpected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << otherexpected << " "
               "Received: " << myothervalue);
  }
}

IMPLEMENT_TEST(config_value_spaces)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("my value with spaces");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  sprokit::config::key_t const mytabkey = sprokit::config::key_t("myblock:mytabs");
  sprokit::config::value_t const mytabvalue = conf->get_value<sprokit::config::value_t>(mytabkey);
  sprokit::config::value_t const tabexpected = sprokit::config::value_t("my	value	with	tabs");

  if (mytabvalue != tabexpected)
  {
    TEST_ERROR("Configuration was not correct: "
               "Expected: " << tabexpected << " "
               "Received: " << mytabvalue);
  }
}

IMPLEMENT_TEST(config_overrides)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myothervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_read_only)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const config = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const rokey = sprokit::config::key_t("myblock:mykey");

  if (!config->is_read_only(rokey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_not_a_flag)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::unrecognized_config_flag_exception,
                   sprokit::extract_configuration(blocks),
                   "using an unknown flag");
}

IMPLEMENT_TEST(config_read_only_override)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::set_on_read_only_value_exception,
                   sprokit::extract_configuration(blocks),
                   "setting a read-only value");
}

IMPLEMENT_TEST(config_append)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_ro)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  if (!conf->is_read_only(mykey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_append_provided)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_provided_ro)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  if (!conf->is_read_only(mykey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_append_comma)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue,othervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended with a comma separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_comma_empty)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("othervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was created with a comma separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_space)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue othervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended with a space separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_space_empty)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("othervalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was created with a space separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_path)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::path_t const expected_path = sprokit::path_t("myvalue") / sprokit::path_t("othervalue");
  sprokit::config::value_t const expected = expected_path.string<sprokit::config::value_t>();

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended with a path separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_path_empty)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::path_t const expected_path = sprokit::path_t(".") / sprokit::path_t("othervalue");
  sprokit::config::value_t const expected = expected_path.string<sprokit::config::value_t>();

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not created properly with a path separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_flag_mismatch_ac)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::config_flag_mismatch_exception,
                   sprokit::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_append_flag_mismatch_ap)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::config_flag_mismatch_exception,
                   sprokit::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_append_flag_mismatch_cp)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::config_flag_mismatch_exception,
                   sprokit::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_append_flag_mismatch_all)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::config_flag_mismatch_exception,
                   sprokit::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_provider_conf)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myotherblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_provider_conf_dep)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const conf = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myotherblock:mykey");
  sprokit::config::value_t const myvalue = conf->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("myvalue");

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }

  sprokit::config::key_t const mymidkey = sprokit::config::key_t("mymidblock:mykey");
  sprokit::config::value_t const mymidvalue = conf->get_value<sprokit::config::value_t>(mymidkey);

  if (mymidvalue != expected)
  {
    TEST_ERROR("Configuration was not overridden: "
               "Expected: " << expected << " "
               "Received: " << mymidvalue);
  }
}

IMPLEMENT_TEST(config_provider_conf_circular_dep)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::circular_config_provide_exception,
                   sprokit::extract_configuration(blocks),
                   "circular configuration provides exist");
}

IMPLEMENT_TEST(config_provider_env)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const config = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const mykey = sprokit::config::key_t("myblock:myenv");
  sprokit::config::value_t const value = config->get_value<sprokit::config::value_t>(mykey);
  sprokit::config::value_t const expected = sprokit::config::value_t("expected");

  if (value != expected)
  {
    TEST_ERROR("Environment was not read properly: "
               "Expected: " << expected << " "
               "Received: " << value);
  }
}

IMPLEMENT_TEST(config_provider_read_only)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::config_t const config = sprokit::extract_configuration(blocks);

  sprokit::config::key_t const rokey = sprokit::config::key_t("myblock:mykey");

  if (!config->is_read_only(rokey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_provider_read_only_override)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::set_on_read_only_value_exception,
                   sprokit::extract_configuration(blocks),
                   "setting a read-only provided value");
}

IMPLEMENT_TEST(config_provider_unprovided)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::unrecognized_provider_exception,
                   sprokit::extract_configuration(blocks),
                   "using an unknown provider");
}

IMPLEMENT_TEST(pipeline_multiplier)
{
  sprokit::pipe_blocks const blocks = sprokit::load_pipe_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  sprokit::pipeline_t const pipeline = sprokit::bake_pipe_blocks(blocks);

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

IMPLEMENT_TEST(cluster_multiplier)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks(blocks);

  sprokit::process_ctor_t const ctor = info->ctor;

  sprokit::config_t const config = sprokit::config::empty_config();

  config->set_value("factor", boost::lexical_cast<sprokit::config::value_t>(30));

  sprokit::process_t const proc = ctor(config);

  sprokit::process_cluster_t const cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(proc);

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

IMPLEMENT_TEST(cluster_missing_cluster)
{
  (void)pipe_file;

  sprokit::cluster_blocks blocks;

  sprokit::process_pipe_block const pipe_block = sprokit::process_pipe_block();
  sprokit::cluster_block const block = pipe_block;

  blocks.push_back(block);

  EXPECT_EXCEPTION(sprokit::missing_cluster_block_exception,
                   sprokit::bake_cluster_blocks(blocks),
                   "baking a set of cluster blocks without a cluster");
}

IMPLEMENT_TEST(cluster_missing_processes)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::cluster_without_processes_exception,
                   sprokit::bake_cluster_blocks(blocks),
                   "baking a cluster without processes");
}

IMPLEMENT_TEST(cluster_missing_ports)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(sprokit::cluster_without_ports_exception,
                   sprokit::bake_cluster_blocks(blocks),
                   "baking a cluster without ports");
}

IMPLEMENT_TEST(cluster_multiple_cluster)
{
  (void)pipe_file;

  sprokit::cluster_blocks blocks;

  sprokit::cluster_pipe_block const cluster_pipe_block = sprokit::cluster_pipe_block();
  sprokit::cluster_block const cluster_block = cluster_pipe_block;

  blocks.push_back(cluster_block);
  blocks.push_back(cluster_block);

  EXPECT_EXCEPTION(sprokit::multiple_cluster_blocks_exception,
                   sprokit::bake_cluster_blocks(blocks),
                   "baking a set of cluster blocks without multiple clusters");
}

IMPLEMENT_TEST(cluster_duplicate_input)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  EXPECT_EXCEPTION(sprokit::duplicate_cluster_input_port_exception,
                   sprokit::bake_cluster_blocks(blocks),
                   "baking a cluster with duplicate input ports");
}

IMPLEMENT_TEST(cluster_duplicate_output)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  EXPECT_EXCEPTION(sprokit::duplicate_cluster_output_port_exception,
                   sprokit::bake_cluster_blocks(blocks),
                   "baking a cluster with duplicate output ports");
}

static void test_cluster(sprokit::process_t const& cluster, std::string const& path);

IMPLEMENT_TEST(cluster_configuration_default)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks(blocks);

  sprokit::process_ctor_t const ctor = info->ctor;

  sprokit::config_t const config = sprokit::config::empty_config();

  sprokit::process_t const cluster = ctor(config);

  std::string const output_path = "test-pipe_bakery-configuration_default.txt";

  test_cluster(cluster, output_path);
}

IMPLEMENT_TEST(cluster_configuration_provide)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks(blocks);

  sprokit::process_ctor_t const ctor = info->ctor;

  sprokit::config_t const config = sprokit::config::empty_config();

  sprokit::process_t const cluster = ctor(config);

  std::string const output_path = "test-pipe_bakery-configuration_provide.txt";

  test_cluster(cluster, output_path);
}

static sprokit::process_cluster_t setup_map_config_cluster(sprokit::process::name_t const& name, sprokit::path_t const& pipe_file);

IMPLEMENT_TEST(cluster_map_config)
{
  /// \todo This test is poorly designed in that it tests for mapping by
  /// leveraging the fact that only mapped configuration get pushed through when
  /// reconfiguring a cluster.

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("expected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_tunable)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("expected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_redirect)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("expected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_modified)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("expected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_not_read_only)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("unexpected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_only_provided)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("unexpected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_only_conf_provided)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("unexpected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_to_non_process)
{
  /// \todo This test is poorly designed because there's no check that / the
  /// mapping wasn't actually created.

  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::config::key_t const key = sprokit::config::key_t("tunable");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("expected");

  new_conf->set_value(cluster_name + sprokit::config::block_sep + key, tuned_value);

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_map_config_not_from_cluster)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::process_cluster_t const cluster = setup_map_config_cluster(cluster_name, pipe_file);

  if (!cluster)
  {
    TEST_ERROR("A cluster was not created");

    return;
  }

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>(sprokit::config::empty_config());

  pipeline->add_process(cluster);
  pipeline->setup_pipeline();

  sprokit::config_t const new_conf = sprokit::config::empty_config();

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const new_key = sprokit::config::key_t("new_key");
  sprokit::config::value_t const tuned_value = sprokit::config::key_t("expected");

  // Fill a block so that the expect process gets reconfigured to do its check;
  // if the block for it is empty, the check won't happen.
  new_conf->set_value(cluster_name + sprokit::config::block_sep + proc_name + sprokit::config::block_sep + new_key, tuned_value);

  pipeline->reconfigure(new_conf);
}

IMPLEMENT_TEST(cluster_override_mapped)
{
  sprokit::process::name_t const cluster_name = sprokit::process::name_t("cluster");

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::process::name_t const proc_name = sprokit::process::name_t("expect");
  sprokit::config::key_t const key = sprokit::config::key_t("tunable");

  sprokit::config::key_t const full_key = proc_name + sprokit::config::block_sep + key;
  sprokit::config::value_t const value = sprokit::config::value_t("unexpected");

  // The problem here is that the expect:tunable parameter is meant to be mapped
  // with the map_config call within the cluster. Due to the implementation,
  // the lines which cause the linking are removed, so if we try to sneak in an
  // invalid parameter, we should get an error about overriding a read-only
  // variable.
  conf->set_value(full_key, value);

  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks(blocks);

  sprokit::process_ctor_t const ctor = info->ctor;

  EXPECT_EXCEPTION(sprokit::set_on_read_only_value_exception,
                   ctor(conf),
                   "manually setting a parameter which is mapped in a cluster");
}

static sprokit::process_t create_process(sprokit::process::type_t const& type, sprokit::process::name_t const& name, sprokit::config_t config = sprokit::config::empty_config());
static sprokit::pipeline_t create_pipeline();

void
test_cluster(sprokit::process_t const& cluster, std::string const& path)
{
  sprokit::process::type_t const proc_typeu = sprokit::process::type_t("numbers");
  sprokit::process::type_t const proc_typet = sprokit::process::type_t("print_number");

  sprokit::process::name_t const proc_nameu = sprokit::process::name_t("upstream");
  sprokit::process::name_t const proc_named = cluster->name();
  sprokit::process::name_t const proc_namet = sprokit::process::name_t("terminal");

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  {
    sprokit::config_t const configu = sprokit::config::empty_config();

    sprokit::config::key_t const start_key = sprokit::config::key_t("start");
    sprokit::config::key_t const end_key = sprokit::config::key_t("end");

    sprokit::config::value_t const start_num = boost::lexical_cast<sprokit::config::value_t>(start_value);
    sprokit::config::value_t const end_num = boost::lexical_cast<sprokit::config::value_t>(end_value);

    configu->set_value(start_key, start_num);
    configu->set_value(end_key, end_num);

    sprokit::config_t const configt = sprokit::config::empty_config();

    sprokit::config::key_t const output_key = sprokit::config::key_t("output");
    sprokit::config::value_t const output_value = sprokit::config::value_t(path);

    configt->set_value(output_key, output_value);

    sprokit::process_t const processu = create_process(proc_typeu, proc_nameu, configu);
    sprokit::process_t const processt = create_process(proc_typet, proc_namet, configt);

    sprokit::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process(processu);
    pipeline->add_process(cluster);
    pipeline->add_process(processt);

    sprokit::process::port_t const port_nameu = sprokit::process::port_t("number");
    sprokit::process::port_t const port_namedi = sprokit::process::port_t("factor");
    sprokit::process::port_t const port_namedo = sprokit::process::port_t("product");
    sprokit::process::port_t const port_namet = sprokit::process::port_t("number");

    pipeline->connect(proc_nameu, port_nameu,
                      proc_named, port_namedi);
    pipeline->connect(proc_named, port_namedo,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

    sprokit::scheduler_t const scheduler = reg->create_scheduler(sprokit::scheduler_registry::default_type, pipeline);

    scheduler->start();
    scheduler->wait();
  }

  std::ifstream fin(path.c_str());

  if (!fin.good())
  {
    TEST_ERROR("Could not open the output file");
  }

  std::string line;

  // From the input cluster.
  static const int32_t factor = 20;

  for (int32_t i = start_value; i < end_value; ++i)
  {
    std::getline(fin, line);

    if (sprokit::config::value_t(line) != boost::lexical_cast<sprokit::config::value_t>(i * factor))
    {
      TEST_ERROR("Did not get expected value: "
                 "Expected: " << i * factor << " "
                 "Received: " << line);
    }
  }

  std::getline(fin, line);

  if (!line.empty())
  {
    TEST_ERROR("Empty line missing");
  }

  if (!fin.eof())
  {
    TEST_ERROR("Not at end of file");
  }
}

sprokit::process_t
create_process(sprokit::process::type_t const& type, sprokit::process::name_t const& name, sprokit::config_t config)
{
  static bool const modules_loaded = (sprokit::load_known_modules(), true);
  static sprokit::process_registry_t const reg = sprokit::process_registry::self();

  (void)modules_loaded;

  return reg->create_process(type, name, config);
}

sprokit::pipeline_t
create_pipeline()
{
  return boost::make_shared<sprokit::pipeline>();
}

sprokit::process_cluster_t
setup_map_config_cluster(sprokit::process::name_t const& name, sprokit::path_t const& pipe_file)
{
  sprokit::cluster_blocks const blocks = sprokit::load_cluster_blocks_from_file(pipe_file);

  sprokit::load_known_modules();

  sprokit::cluster_info_t const info = sprokit::bake_cluster_blocks(blocks);

  sprokit::process_ctor_t const ctor = info->ctor;

  sprokit::config_t const config = sprokit::config::empty_config();

  config->set_value(sprokit::process::config_name, name);

  sprokit::process_t const proc = ctor(config);

  sprokit::process_cluster_t const cluster = boost::dynamic_pointer_cast<sprokit::process_cluster>(proc);

  return cluster;
}
