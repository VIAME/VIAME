/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline_util/load_pipe.h>
#include <vistk/pipeline_util/load_pipe_exception.h>
#include <vistk/pipeline_util/path.h>
#include <vistk/pipeline_util/pipe_declaration_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>

#include <boost/variant.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

#include <cstddef>

#define TEST_ARGS (vistk::path_t const& pipe_file)

DECLARE_TEST(empty);
DECLARE_TEST(comments);
DECLARE_TEST(empty_config);
DECLARE_TEST(config_block);
DECLARE_TEST(config_block_notalnum);
DECLARE_TEST(config_value_spaces);
DECLARE_TEST(one_process);
DECLARE_TEST(connected_processes);
DECLARE_TEST(connected_processes_notalnum);
DECLARE_TEST(include);
DECLARE_TEST(no_exist);
DECLARE_TEST(not_a_file);
DECLARE_TEST(include_no_exist);
DECLARE_TEST(include_not_a_file);
DECLARE_TEST(no_parse);
DECLARE_TEST(parse_error);
DECLARE_TEST(envvar);
DECLARE_TEST(cluster_declare);
DECLARE_TEST(cluster_config);
DECLARE_TEST(cluster_input_map);
DECLARE_TEST(cluster_input_multi_map);
DECLARE_TEST(cluster_output_map);
DECLARE_TEST(cluster_mappings);
DECLARE_TEST(cluster_all);
DECLARE_TEST(cluster_missing_config_description);
DECLARE_TEST(cluster_missing_input_description);
DECLARE_TEST(cluster_missing_output_description);
DECLARE_TEST(cluster_missing_type);
DECLARE_TEST(cluster_missing_type_description);
DECLARE_TEST(cluster_multiple_clusters);
DECLARE_TEST(cluster_not_first);
DECLARE_TEST(cluster_with_slash);
DECLARE_TEST(cluster_input_map_with_slash);
DECLARE_TEST(cluster_output_map_with_slash);
DECLARE_TEST(process_with_slash);
DECLARE_TEST(connect_input_with_slash);
DECLARE_TEST(connect_output_with_slash);

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  testname_t const testname = argv[1];
  vistk::path_t const pipe_dir = argv[2];

  vistk::path_t const pipe_file = pipe_dir / (testname + pipe_ext);

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, empty);
  ADD_TEST(tests, comments);
  ADD_TEST(tests, empty_config);
  ADD_TEST(tests, config_block);
  ADD_TEST(tests, config_block_notalnum);
  ADD_TEST(tests, config_value_spaces);
  ADD_TEST(tests, one_process);
  ADD_TEST(tests, connected_processes);
  ADD_TEST(tests, connected_processes_notalnum);
  ADD_TEST(tests, include);
  ADD_TEST(tests, no_exist);
  ADD_TEST(tests, not_a_file);
  ADD_TEST(tests, include_no_exist);
  ADD_TEST(tests, include_not_a_file);
  ADD_TEST(tests, no_parse);
  ADD_TEST(tests, parse_error);
  ADD_TEST(tests, envvar);
  ADD_TEST(tests, cluster_declare);
  ADD_TEST(tests, cluster_config);
  ADD_TEST(tests, cluster_input_map);
  ADD_TEST(tests, cluster_input_multi_map);
  ADD_TEST(tests, cluster_output_map);
  ADD_TEST(tests, cluster_mappings);
  ADD_TEST(tests, cluster_all);
  ADD_TEST(tests, cluster_missing_config_description);
  ADD_TEST(tests, cluster_missing_input_description);
  ADD_TEST(tests, cluster_missing_output_description);
  ADD_TEST(tests, cluster_missing_type);
  ADD_TEST(tests, cluster_missing_type_description);
  ADD_TEST(tests, cluster_multiple_clusters);
  ADD_TEST(tests, cluster_not_first);
  ADD_TEST(tests, cluster_with_slash);
  ADD_TEST(tests, cluster_input_map_with_slash);
  ADD_TEST(tests, cluster_output_map_with_slash);
  ADD_TEST(tests, process_with_slash);
  ADD_TEST(tests, connect_input_with_slash);
  ADD_TEST(tests, connect_output_with_slash);

  RUN_TEST(tests, testname, pipe_file);
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

IMPLEMENT_TEST(empty)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

IMPLEMENT_TEST(comments)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

IMPLEMENT_TEST(empty_config)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

IMPLEMENT_TEST(config_block)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

IMPLEMENT_TEST(config_block_notalnum)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(2, 0, 0, 0);
}

IMPLEMENT_TEST(config_value_spaces)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

IMPLEMENT_TEST(one_process)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 1, 0, 0);
}

IMPLEMENT_TEST(connected_processes)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 2, 1, 0);
}

IMPLEMENT_TEST(connected_processes_notalnum)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 2, 3, 0);
}

IMPLEMENT_TEST(include)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(1, 0, 0, 0);
}

IMPLEMENT_TEST(no_exist)
{
  EXPECT_EXCEPTION(vistk::file_no_exist_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading a non-existent file");
}

IMPLEMENT_TEST(not_a_file)
{
  EXPECT_EXCEPTION(vistk::not_a_file_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading a non-file");
}

IMPLEMENT_TEST(include_no_exist)
{
  EXPECT_EXCEPTION(vistk::file_no_exist_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "including a non-existent file");
}

IMPLEMENT_TEST(include_not_a_file)
{
  EXPECT_EXCEPTION(vistk::not_a_file_exception,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "including a non-file");
}

IMPLEMENT_TEST(no_parse)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "loading an invalid file");
}

IMPLEMENT_TEST(parse_error)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(envvar)
{
  (void)pipe_file;

  std::stringstream sstr;

  sstr << "!include include_test.pipe" << std::endl;

  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks(sstr);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 0);
}

IMPLEMENT_TEST(cluster_declare)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_config)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_input_map)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_input_multi_map)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_output_map)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_mappings)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_all)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_missing_config_description)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_missing_input_description)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_missing_output_description)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_missing_type)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_missing_type_description)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_multiple_clusters)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_not_first)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_with_slash)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(cluster_input_map_with_slash)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(cluster_output_map_with_slash)
{
  vistk::cluster_blocks const blocks = vistk::load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 0, 1);
}

IMPLEMENT_TEST(process_with_slash)
{
  EXPECT_EXCEPTION(vistk::failed_to_parse,
                   vistk::load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

IMPLEMENT_TEST(connect_input_with_slash)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 1, 0);
}

IMPLEMENT_TEST(connect_output_with_slash)
{
  vistk::pipe_blocks const blocks = vistk::load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  std::for_each(blocks.begin(), blocks.end(), boost::apply_visitor(v));

  v.expect(0, 0, 1, 0);
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
