/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline_util/pipe_declaration_types.h>
#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/load_pipe_exception.h>
#include <vistk/pipeline_util/pipe_bakery.h>
#include <vistk/pipeline_util/pipe_bakery_exception.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>

#include <boost/filesystem/path.hpp>
#include <boost/variant.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

static std::string const pipe_ext = ".pipe";

static void run_test(std::string const& test_name, boost::filesystem::path const& pipe_file);

int
main(int argc, char* argv[])
{
  if (argc != 3)
  {
    TEST_ERROR("Expected two arguments");

    return 1;
  }

  vistk::load_known_modules();

  std::string const test_name = argv[1];
  boost::filesystem::path const pipe_dir = argv[2];

  boost::filesystem::path const pipe_file = pipe_dir / (test_name + pipe_ext);

  try
  {
    run_test(test_name, pipe_file);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return 1;
  }

  return 0;
}

static void test_empty(boost::filesystem::path const& pipe_file);
static void test_comments(boost::filesystem::path const& pipe_file);
static void test_empty_config(boost::filesystem::path const& pipe_file);
static void test_config_block(boost::filesystem::path const& pipe_file);
static void test_one_process(boost::filesystem::path const& pipe_file);
static void test_connected_processes(boost::filesystem::path const& pipe_file);
static void test_config_overrides(boost::filesystem::path const& pipe_file);
static void test_config_read_only(boost::filesystem::path const& pipe_file);
static void test_config_not_a_flag(boost::filesystem::path const& pipe_file);
static void test_config_read_only_override(boost::filesystem::path const& pipe_file);
static void test_config_provider_conf(boost::filesystem::path const& pipe_file);
static void test_config_provider_conf_dep(boost::filesystem::path const& pipe_file);
static void test_config_provider_conf_circular_dep(boost::filesystem::path const& pipe_file);
static void test_config_provider_env(boost::filesystem::path const& pipe_file);
static void test_config_provider_read_only(boost::filesystem::path const& pipe_file);
static void test_config_provider_read_only_override(boost::filesystem::path const& pipe_file);
static void test_config_provider_unprovided(boost::filesystem::path const& pipe_file);
static void test_include(boost::filesystem::path const& pipe_file);
static void test_no_exist(boost::filesystem::path const& pipe_file);
static void test_not_a_file(boost::filesystem::path const& pipe_file);
static void test_include_no_exist(boost::filesystem::path const& pipe_file);
static void test_include_not_a_file(boost::filesystem::path const& pipe_file);
static void test_group_declare(boost::filesystem::path const& pipe_file);
static void test_group_config(boost::filesystem::path const& pipe_file);
static void test_group_input_map(boost::filesystem::path const& pipe_file);
static void test_group_output_map(boost::filesystem::path const& pipe_file);
static void test_group_input_map_flags(boost::filesystem::path const& pipe_file);
static void test_group_output_map_flags(boost::filesystem::path const& pipe_file);
static void test_group_mappings(boost::filesystem::path const& pipe_file);
static void test_group_all(boost::filesystem::path const& pipe_file);
static void test_no_parse(boost::filesystem::path const& pipe_file);
static void test_parse_error(boost::filesystem::path const& pipe_file);

void
run_test(std::string const& test_name, boost::filesystem::path const& pipe_file)
{
  if (test_name == "empty")
  {
    test_empty(pipe_file);
  }
  else if (test_name == "comments")
  {
    test_comments(pipe_file);
  }
  else if (test_name == "empty_config")
  {
    test_empty_config(pipe_file);
  }
  else if (test_name == "config_block")
  {
    test_config_block(pipe_file);
  }
  else if (test_name == "one_process")
  {
    test_one_process(pipe_file);
  }
  else if (test_name == "connected_processes")
  {
    test_connected_processes(pipe_file);
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
  else if (test_name == "include")
  {
    test_include(pipe_file);
  }
  else if (test_name == "no_exist")
  {
    test_no_exist(pipe_file);
  }
  else if (test_name == "not_a_file")
  {
    test_not_a_file(pipe_file);
  }
  else if (test_name == "include_no_exist")
  {
    test_include_no_exist(pipe_file);
  }
  else if (test_name == "include_not_a_file")
  {
    test_include_not_a_file(pipe_file);
  }
  else if (test_name == "group_declare")
  {
    test_group_declare(pipe_file);
  }
  else if (test_name == "group_config")
  {
    test_group_config(pipe_file);
  }
  else if (test_name == "group_input_map")
  {
    test_group_input_map(pipe_file);
  }
  else if (test_name == "group_output_map")
  {
    test_group_output_map(pipe_file);
  }
  else if (test_name == "group_input_map_flags")
  {
    test_group_input_map_flags(pipe_file);
  }
  else if (test_name == "group_output_map_flags")
  {
    test_group_output_map_flags(pipe_file);
  }
  else if (test_name == "group_mappings")
  {
    test_group_mappings(pipe_file);
  }
  else if (test_name == "group_all")
  {
    test_group_all(pipe_file);
  }
  else if (test_name == "no_parse")
  {
    test_no_parse(pipe_file);
  }
  else if (test_name == "parse_error")
  {
    test_parse_error(pipe_file);
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

class test_visitor
  : public boost::static_visitor<>
{
  public:
    typedef enum
    {
      CONFIG_BLOCK,
      PROCESS_BLOCK,
      CONNECT_BLOCK,
      GROUP_BLOCK
    } block_type_t;

    typedef std::vector<block_type_t> block_types_t;

    test_visitor();
    ~test_visitor();

    void operator () (vistk::config_pipe_block const& config_block);
    void operator () (vistk::process_pipe_block const& process_block);
    void operator () (vistk::connect_pipe_block const& connect_block);
    void operator () (vistk::group_pipe_block const& group_block);

    void expect(size_t config_expect,
                size_t process_expect,
                size_t connect_expect,
                size_t group_expect) const;
    void output_report() const;

    static void print_char(block_type_t type);

    size_t config_count;
    size_t process_count;
    size_t connect_count;
    size_t group_count;

    size_t total_count;

    block_types_t types;
};

void
test_empty(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

void
test_comments(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

void
test_empty_config(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

void
test_config_block(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = conf->get_value<vistk::config::key_t>("myblock:mykey");
  vistk::config::key_t const expected = vistk::config::key_t("myvalue");

  if (mykey != expected)
  {
    TEST_ERROR("Configuration was not overriden: "
               "Expected: " << expected << " "
               "Received: " << mykey);
  }
}

void
test_one_process(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 1, 0, 0);
}

void
test_connected_processes(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 2, 1, 0);
}

void
test_config_overrides(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = conf->get_value<vistk::config::key_t>("myblock:mykey");
  vistk::config::key_t const expected = vistk::config::key_t("myothervalue");

  if (mykey != expected)
  {
    TEST_ERROR("Configuration was not overriden: "
               "Expected: " << expected << " "
               "Received: " << mykey);
  }
}

void
test_config_read_only(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

void
test_config_not_a_flag(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);

  EXPECT_EXCEPTION(vistk::unrecognized_config_flag_exception,
                   vistk::extract_configuration(blocks),
                   "using an unknown flag");
}

void
test_config_read_only_override(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   vistk::extract_configuration(blocks),
                   "setting a read-only value");
}

void
test_config_provider_conf(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = conf->get_value<vistk::config::key_t>("myotherblock:mykey");
  vistk::config::key_t const expected = vistk::config::key_t("myvalue");

  if (mykey != expected)
  {
    TEST_ERROR("Configuration was not overriden: "
               "Expected: " << expected << " "
               "Received: " << mykey);
  }
}

void
test_config_provider_conf_dep(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(3, 0, 0, 0);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = conf->get_value<vistk::config::key_t>("myotherblock:mykey");
  vistk::config::key_t const expected = vistk::config::key_t("myvalue");

  if (mykey != expected)
  {
    TEST_ERROR("Configuration was not overriden: "
               "Expected: " << expected << " "
               "Received: " << mykey);
  }

  vistk::config::key_t const mymidkey = conf->get_value<vistk::config::key_t>("mymidblock:mykey");

  if (mykey != expected)
  {
    TEST_ERROR("Configuration was not overriden: "
               "Expected: " << expected << " "
               "Received: " << mykey);
  }
}

void
test_config_provider_conf_circular_dep(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);

  EXPECT_EXCEPTION(vistk::circular_config_provide_exception,
                   vistk::extract_configuration(blocks),
                   "circular configuration provides exist");
}

void
test_config_provider_env(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);

  vistk::extract_configuration(blocks);
}

void
test_config_provider_read_only(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);

  vistk::extract_configuration(blocks);
}

void
test_config_provider_read_only_override(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);

  EXPECT_EXCEPTION(vistk::set_on_read_only_value_exception,
                   vistk::extract_configuration(blocks),
                   "setting a read-only provided value");
}

void
test_config_provider_unprovided(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);

  EXPECT_EXCEPTION(vistk::unrecognized_provider_exception,
                   vistk::extract_configuration(blocks),
                   "using an unknown provider");
}

void
test_include(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);

  vistk::config_t const conf = vistk::extract_configuration(blocks);

  vistk::config::key_t const mykey = conf->get_value<vistk::config::key_t>("myblock:mykey");
  vistk::config::key_t const expected = vistk::config::key_t("myvalue");

  if (mykey != expected)
  {
    TEST_ERROR("Configuration was not overriden: "
               "Expected: " << expected << " "
               "Received: " << mykey);
  }
}

void
test_no_exist(boost::filesystem::path const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::file_no_exist_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading a non-existent file");
}

void
test_not_a_file(boost::filesystem::path const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::not_a_file_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading a non-file");
}

void
test_include_no_exist(boost::filesystem::path const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::file_no_exist_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "including a non-existent file");
}

void
test_include_not_a_file(boost::filesystem::path const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::not_a_file_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "including a non-file");
}

void
test_group_declare(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_config(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_input_map(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_output_map(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_input_map_flags(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_output_map_flags(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_mappings(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_group_all(boost::filesystem::path const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_no_parse(boost::filesystem::path const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading an invalid file");
}

void
test_parse_error(boost::filesystem::path const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

test_visitor
::test_visitor()
  : config_count(0)
  , process_count(0)
  , connect_count(0)
  , group_count(0)
  , total_count(0)
{
}

test_visitor
::~test_visitor()
{
}

void
test_visitor
::operator () (vistk::config_pipe_block const& /*config_block*/)
{
  ++config_count;
  ++total_count;

  types.push_back(CONFIG_BLOCK);
}

void
test_visitor
::operator () (vistk::process_pipe_block const& /*process_block*/)
{
  ++process_count;
  ++total_count;

  types.push_back(PROCESS_BLOCK);
}

void
test_visitor
::operator () (vistk::connect_pipe_block const& /*connect_block*/)
{
  ++connect_count;
  ++total_count;

  types.push_back(CONNECT_BLOCK);
}

void
test_visitor
::operator () (vistk::group_pipe_block const& /*group_block*/)
{
  ++group_count;
  ++total_count;

  types.push_back(GROUP_BLOCK);
}

void
test_visitor
::expect(size_t config_expect,
         size_t process_expect,
         size_t connect_expect,
         size_t group_expect) const
{
  bool is_good = true;

  if (config_expect != config_count)
  {
    TEST_ERROR("config count: "
               "Expected: " << config_expect << " "
               "Received: " << config_count);
    is_good = false;
  }
  if (process_expect != process_count)
  {
    TEST_ERROR("process count: "
               "Expected: " << process_expect << " "
               "Received: " << process_count);
    is_good = false;
  }
  if (connect_expect != connect_count)
  {
    TEST_ERROR("connect count: "
               "Expected: " << connect_expect << " "
               "Received: " << connect_count);
    is_good = false;
  }
  if (group_expect != group_count)
  {
    TEST_ERROR("group count: "
               "Expected: " << group_expect << " "
               "Received: " << group_count);
    is_good = false;
  }

  if (!is_good)
  {
    output_report();
  }
}

void
test_visitor
::output_report() const
{
  std::cerr << "Total blocks  : " << total_count << std::endl;
  std::cerr << "config blocks : " << config_count << std::endl;
  std::cerr << "process blocks: " << process_count << std::endl;
  std::cerr << "connect blocks: " << connect_count << std::endl;
  std::cerr << "group blocks  : " << group_count << std::endl;

  std::cerr << "Order: ";

  std::for_each(types.begin(), types.end(), print_char);

  std::cerr << std::endl;
}

void
test_visitor
::print_char(block_type_t type)
{
  char c;

  switch (type)
  {
    case CONFIG_BLOCK:
      c = 'C';
      break;
    case PROCESS_BLOCK:
      c = 'p';
      break;
    case CONNECT_BLOCK:
      c = 'c';
      break;
    case GROUP_BLOCK:
      c = 'g';
      break;
    default:
      c = 'U';
      break;
  }

  std::cerr << c;
}
