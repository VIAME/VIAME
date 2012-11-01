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

  vistk::extract_configuration(blocks);
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
