/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline_util/pipe_declaration_types.h>
#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/load_pipe_exception.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>

#include <boost/variant.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cstddef>
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

  vistk::load_known_modules();

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

static void test_empty(vistk::path_t const& pipe_file);
static void test_comments(vistk::path_t const& pipe_file);
static void test_empty_config(vistk::path_t const& pipe_file);
static void test_config_block(vistk::path_t const& pipe_file);
static void test_config_block_notalnum(vistk::path_t const& pipe_file);
static void test_config_value_spaces(vistk::path_t const& pipe_file);
static void test_one_process(vistk::path_t const& pipe_file);
static void test_connected_processes(vistk::path_t const& pipe_file);
static void test_connected_processes_notalnum(vistk::path_t const& pipe_file);
static void test_include(vistk::path_t const& pipe_file);
static void test_no_exist(vistk::path_t const& pipe_file);
static void test_not_a_file(vistk::path_t const& pipe_file);
static void test_include_no_exist(vistk::path_t const& pipe_file);
static void test_include_not_a_file(vistk::path_t const& pipe_file);
static void test_no_parse(vistk::path_t const& pipe_file);
static void test_parse_error(vistk::path_t const& pipe_file);
static void test_envvar(vistk::path_t const& pipe_file);
static void test_cluster_declare(vistk::path_t const& pipe_file);
static void test_cluster_config(vistk::path_t const& pipe_file);
static void test_cluster_input_map(vistk::path_t const& pipe_file);
static void test_cluster_input_multi_map(vistk::path_t const& pipe_file);
static void test_cluster_output_map(vistk::path_t const& pipe_file);
static void test_cluster_mappings(vistk::path_t const& pipe_file);
static void test_cluster_all(vistk::path_t const& pipe_file);
static void test_cluster_missing_config_description(vistk::path_t const& pipe_file);
static void test_cluster_missing_input_description(vistk::path_t const& pipe_file);
static void test_cluster_missing_output_description(vistk::path_t const& pipe_file);
static void test_cluster_missing_type(vistk::path_t const& pipe_file);
static void test_cluster_missing_type_description(vistk::path_t const& pipe_file);
static void test_cluster_multiple_clusters(vistk::path_t const& pipe_file);
static void test_cluster_not_first(vistk::path_t const& pipe_file);

void
run_test(std::string const& test_name, vistk::path_t const& pipe_file)
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
  else if (test_name == "config_block_notalnum")
  {
    test_config_block_notalnum(pipe_file);
  }
  else if (test_name == "config_value_spaces")
  {
    test_config_value_spaces(pipe_file);
  }
  else if (test_name == "one_process")
  {
    test_one_process(pipe_file);
  }
  else if (test_name == "connected_processes")
  {
    test_connected_processes(pipe_file);
  }
  else if (test_name == "connected_processes_notalnum")
  {
    test_connected_processes_notalnum(pipe_file);
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
  else if (test_name == "no_parse")
  {
    test_no_parse(pipe_file);
  }
  else if (test_name == "parse_error")
  {
    test_parse_error(pipe_file);
  }
  else if (test_name == "envvar")
  {
    test_envvar(pipe_file);
  }
  else if (test_name == "cluster_declare")
  {
    test_cluster_declare(pipe_file);
  }
  else if (test_name == "cluster_config")
  {
    test_cluster_config(pipe_file);
  }
  else if (test_name == "cluster_input_map")
  {
    test_cluster_input_map(pipe_file);
  }
  else if (test_name == "cluster_input_multi_map")
  {
    test_cluster_input_multi_map(pipe_file);
  }
  else if (test_name == "cluster_output_map")
  {
    test_cluster_output_map(pipe_file);
  }
  else if (test_name == "cluster_mappings")
  {
    test_cluster_mappings(pipe_file);
  }
  else if (test_name == "cluster_all")
  {
    test_cluster_all(pipe_file);
  }
  else if (test_name == "cluster_missing_config_description")
  {
    test_cluster_missing_config_description(pipe_file);
  }
  else if (test_name == "cluster_missing_input_description")
  {
    test_cluster_missing_input_description(pipe_file);
  }
  else if (test_name == "cluster_missing_output_description")
  {
    test_cluster_missing_output_description(pipe_file);
  }
  else if (test_name == "cluster_missing_type")
  {
    test_cluster_missing_type(pipe_file);
  }
  else if (test_name == "cluster_missing_type_description")
  {
    test_cluster_missing_type_description(pipe_file);
  }
  else if (test_name == "cluster_multiple_clusters")
  {
    test_cluster_multiple_clusters(pipe_file);
  }
  else if (test_name == "cluster_not_first")
  {
    test_cluster_not_first(pipe_file);
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
      CLUSTER_BLOCK
    } block_type_t;

    typedef std::vector<block_type_t> block_types_t;

    test_visitor();
    ~test_visitor();

    void operator () (vistk::config_pipe_block const& config_block);
    void operator () (vistk::process_pipe_block const& process_block);
    void operator () (vistk::connect_pipe_block const& connect_block);
    void operator () (vistk::cluster_pipe_block const& cluster_block);

    void expect(size_t config_expect,
                size_t process_expect,
                size_t connect_expect,
                size_t cluster_expect) const;
    void output_report() const;

    static void print_char(block_type_t type);

    size_t config_count;
    size_t process_count;
    size_t connect_count;
    size_t cluster_count;

    size_t total_count;

    block_types_t types;
};

void
test_empty(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

void
test_comments(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

void
test_empty_config(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

void
test_config_block(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

void
test_config_block_notalnum(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);
}

void
test_config_value_spaces(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

void
test_one_process(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 1, 0, 0);
}

void
test_connected_processes(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 2, 1, 0);
}

void
test_connected_processes_notalnum(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 2, 3, 0);
}

void
test_include(vistk::path_t const& pipe_file)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

void
test_no_exist(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::file_no_exist_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading a non-existent file");
}

void
test_not_a_file(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::not_a_file_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading a non-file");
}

void
test_include_no_exist(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::file_no_exist_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "including a non-existent file");
}

void
test_include_not_a_file(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::not_a_file_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "including a non-file");
}

void
test_no_parse(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading an invalid file");
}

void
test_parse_error(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_envvar(vistk::path_t const& /*pipe_file*/)
{
  std::stringstream sstr;

  sstr << "!include include_test.pipe" << std::endl;

  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks(sstr);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

void
test_cluster_declare(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_config(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_input_map(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_input_multi_map(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_output_map(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_mappings(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_all(vistk::path_t const& pipe_file)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

void
test_cluster_missing_config_description(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_cluster_missing_input_description(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_cluster_missing_output_description(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_cluster_missing_type(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_cluster_missing_type_description(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_cluster_multiple_clusters(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

void
test_cluster_not_first(vistk::path_t const& pipe_file)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

test_visitor
::test_visitor()
  : config_count(0)
  , process_count(0)
  , connect_count(0)
  , cluster_count(0)
  , total_count(0)
  , types()
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
::operator () (vistk::cluster_pipe_block const& /*cluster_block*/)
{
  ++cluster_count;
  ++total_count;

  types.push_back(CLUSTER_BLOCK);
}

void
test_visitor
::expect(size_t config_expect,
         size_t process_expect,
         size_t connect_expect,
         size_t cluster_expect) const
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
  if (cluster_expect != cluster_count)
  {
    TEST_ERROR("cluster count: "
               "Expected: " << cluster_expect << " "
               "Received: " << cluster_count);
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
  std::cerr << "cluster blocks: " << cluster_count << std::endl;

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
    case CLUSTER_BLOCK:
      c = 'P';
      break;
    default:
      c = 'U';
      break;
  }

  std::cerr << c;
}
