/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
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
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_registry.h>

#include <boost/lexical_cast.hpp>

#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cstdlib>

#define TEST_ARGS (vistk::path_t const& pipe_file)

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
DECLARE_TEST(config_cappend);
DECLARE_TEST(config_cappend_empty);
DECLARE_TEST(config_pappend);
DECLARE_TEST(config_pappend_empty);
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

/// \todo Add tests for clusters without ports or processes.

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  std::string const testname = argv[1];
  vistk::path_t const pipe_dir = argv[2];

  vistk::path_t const pipe_file = pipe_dir / (testname + pipe_ext);

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
  ADD_TEST(tests, config_cappend);
  ADD_TEST(tests, config_cappend_empty);
  ADD_TEST(tests, config_pappend);
  ADD_TEST(tests, config_pappend_empty);
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

  RUN_TEST(tests, testname, pipe_file);
}

IMPLEMENT_TEST(config_block)
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

IMPLEMENT_TEST(config_block_notalnum)
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

IMPLEMENT_TEST(config_value_spaces)
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

IMPLEMENT_TEST(config_overrides)
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

IMPLEMENT_TEST(config_read_only)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const config = vistk::extract_configuration(blocks);

  vistk::config::key_t const rokey = vistk::config::key_t("myblock:mykey");

  if (!config->is_read_only(rokey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_not_a_flag)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::unrecognized_config_flag_exception,
                   vistk::extract_configuration(blocks),
                   "using an unknown flag");
}

IMPLEMENT_TEST(config_read_only_override)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   vistk::extract_configuration(blocks),
                   "setting a read-only value");
}

IMPLEMENT_TEST(config_append)
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

IMPLEMENT_TEST(config_append_ro)
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

  if (!conf->is_read_only(mykey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_append_provided)
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

IMPLEMENT_TEST(config_append_provided_ro)
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

  if (!conf->is_read_only(mykey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_cappend)
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

IMPLEMENT_TEST(config_cappend_empty)
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

IMPLEMENT_TEST(config_pappend)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::path_t const expected_path = vistk::path_t("myvalue") / vistk::path_t("othervalue");
  vistk::config::value_t const expected = expected_path.string<vistk::config::value_t>();

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not appended with a path separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_pappend_empty)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = vistk::config::key_t("myblock:mykey");
  vistk::config::value_t const myvalue = conf->get_value<vistk::config::value_t>(mykey);
  vistk::path_t const expected_path = vistk::path_t(".") / vistk::path_t("othervalue");
  vistk::config::value_t const expected = expected_path.string<vistk::config::value_t>();

  if (myvalue != expected)
  {
    TEST_ERROR("Configuration value was not created properly with a path separator: "
               "Expected: " << expected << " "
               "Received: " << myvalue);
  }
}

IMPLEMENT_TEST(config_append_flag_mismatch_ac)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::config_flag_mismatch_exception,
                   vistk::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_append_flag_mismatch_ap)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::config_flag_mismatch_exception,
                   vistk::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_append_flag_mismatch_cp)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::config_flag_mismatch_exception,
                   vistk::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_append_flag_mismatch_all)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::config_flag_mismatch_exception,
                   vistk::extract_configuration(blocks),
                   "a configuration value has mismatch configuration flags");
}

IMPLEMENT_TEST(config_provider_conf)
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

IMPLEMENT_TEST(config_provider_conf_dep)
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

IMPLEMENT_TEST(config_provider_conf_circular_dep)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::circular_config_provide_exception,
                   vistk::extract_configuration(blocks),
                   "circular configuration provides exist");
}

IMPLEMENT_TEST(config_provider_env)
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

IMPLEMENT_TEST(config_provider_read_only)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  vistk::config_t const config = vistk::extract_configuration(blocks);

  vistk::config::key_t const rokey = vistk::config::key_t("myblock:mykey");

  if (!config->is_read_only(rokey))
  {
    TEST_ERROR("The configuration value was not marked as read only");
  }
}

IMPLEMENT_TEST(config_provider_read_only_override)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   vistk::extract_configuration(blocks),
                   "setting a read-only provided value");
}

IMPLEMENT_TEST(config_provider_unprovided)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::unrecognized_provider_exception,
                   vistk::extract_configuration(blocks),
                   "using an unknown provider");
}

IMPLEMENT_TEST(pipeline_multiplier)
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

IMPLEMENT_TEST(cluster_multiplier)
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

IMPLEMENT_TEST(cluster_missing_cluster)
{
  (void)pipe_file;

  vistk::cluster_blocks blocks;

  vistk::process_pipe_block const pipe_block = vistk::process_pipe_block();
  vistk::cluster_block const block = pipe_block;

  blocks.push_back(block);

  EXPECT_EXCEPTION(vistk::missing_cluster_block_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a set of cluster blocks without a cluster");
}

IMPLEMENT_TEST(cluster_missing_processes)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::cluster_without_processes_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a cluster without processes");
}

IMPLEMENT_TEST(cluster_missing_ports)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  EXPECT_EXCEPTION(vistk::cluster_without_ports_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a cluster without ports");
}

IMPLEMENT_TEST(cluster_multiple_cluster)
{
  (void)pipe_file;

  vistk::cluster_blocks blocks;

  vistk::cluster_pipe_block const cluster_pipe_block = vistk::cluster_pipe_block();
  vistk::cluster_block const cluster_block = cluster_pipe_block;

  blocks.push_back(cluster_block);
  blocks.push_back(cluster_block);

  EXPECT_EXCEPTION(vistk::multiple_cluster_blocks_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a set of cluster blocks without multiple clusters");
}

IMPLEMENT_TEST(cluster_duplicate_input)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  EXPECT_EXCEPTION(vistk::duplicate_cluster_input_port_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a cluster with duplicate input ports");
}

IMPLEMENT_TEST(cluster_duplicate_output)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  EXPECT_EXCEPTION(vistk::duplicate_cluster_output_port_exception,
                   vistk::bake_cluster_blocks(blocks),
                   "baking a cluster with duplicate output ports");
}

static void test_cluster(vistk::process_t const& cluster, std::string const& path);

IMPLEMENT_TEST(cluster_configuration_default)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  vistk::cluster_info_t const info = vistk::bake_cluster_blocks(blocks);

  vistk::process_ctor_t const ctor = info->ctor;

  vistk::config_t const config = vistk::config::empty_config();

  vistk::process_t const cluster = ctor(config);

  std::string const output_path = "test-pipe_bakery-configuration_default.txt";

  test_cluster(cluster, output_path);
}

IMPLEMENT_TEST(cluster_configuration_provide)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  vistk::load_known_modules();

  vistk::cluster_info_t const info = vistk::bake_cluster_blocks(blocks);

  vistk::process_ctor_t const ctor = info->ctor;

  vistk::config_t const config = vistk::config::empty_config();

  vistk::process_t const cluster = ctor(config);

  std::string const output_path = "test-pipe_bakery-configuration_provide.txt";

  test_cluster(cluster, output_path);
}

static vistk::process_t create_process(vistk::process::type_t const& type, vistk::process::name_t const& name, vistk::config_t config = vistk::config::empty_config());
static vistk::pipeline_t create_pipeline();

void
test_cluster(vistk::process_t const& cluster, std::string const& path)
{
  vistk::process::type_t const proc_typeu = vistk::process::type_t("numbers");
  vistk::process::type_t const proc_typet = vistk::process::type_t("print_number");

  vistk::process::name_t const proc_nameu = vistk::process::name_t("upstream");
  vistk::process::name_t const proc_named = cluster->name();
  vistk::process::name_t const proc_namet = vistk::process::name_t("terminal");

  int32_t const start_value = 10;
  int32_t const end_value = 20;

  {
    vistk::config_t const configu = vistk::config::empty_config();

    vistk::config::key_t const start_key = vistk::config::key_t("start");
    vistk::config::key_t const end_key = vistk::config::key_t("end");

    vistk::config::value_t const start_num = boost::lexical_cast<vistk::config::value_t>(start_value);
    vistk::config::value_t const end_num = boost::lexical_cast<vistk::config::value_t>(end_value);

    configu->set_value(start_key, start_num);
    configu->set_value(end_key, end_num);

    vistk::config_t const configt = vistk::config::empty_config();

    vistk::config::key_t const output_key = vistk::config::key_t("output");
    vistk::config::value_t const output_value = vistk::config::value_t(path);

    configt->set_value(output_key, output_value);

    vistk::process_t const processu = create_process(proc_typeu, proc_nameu, configu);
    vistk::process_t const processt = create_process(proc_typet, proc_namet, configt);

    vistk::pipeline_t const pipeline = create_pipeline();

    pipeline->add_process(processu);
    pipeline->add_process(cluster);
    pipeline->add_process(processt);

    vistk::process::port_t const port_nameu = vistk::process::port_t("number");
    vistk::process::port_t const port_namedi = vistk::process::port_t("factor");
    vistk::process::port_t const port_namedo = vistk::process::port_t("product");
    vistk::process::port_t const port_namet = vistk::process::port_t("number");

    pipeline->connect(proc_nameu, port_nameu,
                      proc_named, port_namedi);
    pipeline->connect(proc_named, port_namedo,
                      proc_namet, port_namet);

    pipeline->setup_pipeline();

    vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

    vistk::scheduler_t const scheduler = reg->create_scheduler(vistk::scheduler_registry::default_type, pipeline);

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

    if (vistk::config::value_t(line) != boost::lexical_cast<vistk::config::value_t>(i * factor))
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

vistk::process_t
create_process(vistk::process::type_t const& type, vistk::process::name_t const& name, vistk::config_t config)
{
  static bool const modules_loaded = (vistk::load_known_modules(), true);
  static vistk::process_registry_t const reg = vistk::process_registry::self();

  (void)modules_loaded;

  return reg->create_process(type, name, config);
}

vistk::pipeline_t
create_pipeline()
{
  return boost::make_shared<vistk::pipeline>();
}
