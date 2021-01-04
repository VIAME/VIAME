// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline_util/pipe_declaration_types.h>

#include <vital/config/config_block.h>
#include <kwiversys/SystemTools.hxx>

#include <algorithm>
#include <sstream>
#include <vector>

#include <cstddef>

#define TEST_ARGS (kwiver::vital::path_t const& pipe_file)

DECLARE_TEST_MAP();

static std::string const pipe_ext = ".pipe";

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  testname_t const testname = argv[1];
  kwiver::vital::path_t const pipe_dir = argv[2];

  kwiver::vital::path_t const pipe_file = pipe_dir + "/" +  testname + pipe_ext;

  RUN_TEST(testname, pipe_file);
}

class test_visitor
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

    void operator () (sprokit::config_pipe_block const& config_block);
    void operator () (sprokit::process_pipe_block const& process_block);
    void operator () (sprokit::connect_pipe_block const& connect_block);
    void operator () (sprokit::cluster_pipe_block const& cluster_block);

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

// ----------------------------------------------------------------------------
sprokit::pipe_blocks load_pipe_blocks_from_file( const std::string& file )
{
  sprokit::pipeline_builder builder;
  builder.load_pipeline( file );
  return builder.pipeline_blocks();
}

// ----------------------------------------------------------------------------
sprokit::cluster_blocks load_cluster_blocks_from_file( const std::string& file )
{
  sprokit::pipeline_builder builder;
  builder.load_cluster( file );
  return builder.cluster_blocks();
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(empty)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(comments)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(empty_config)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_block)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_relativepath)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_long_block)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_nested_block)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_unclosed_block)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_unopened_block)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_block_notalnum)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(2, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(config_value_spaces)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(one_process)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 1, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(connected_processes)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 2, 1, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(connected_processes_notalnum)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 2, 3, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(include)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(include_env)
{
  // Supply part of file name
  kwiversys::SystemTools::PutEnv( "FoO=config" );

  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(1, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(no_exist)
{
  EXPECT_EXCEPTION(sprokit::file_no_exist_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "loading a non-existent file");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(include_no_exist)
{
  EXPECT_EXCEPTION(sprokit::file_no_exist_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "including a non-existent file");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(no_parse)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "loading an invalid file");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(parse_error)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
TEST_PROPERTY(ENVIRONMENT, SPROKIT_PIPE_INCLUDE_PATH=@CMAKE_CURRENT_SOURCE_DIR@)
IMPLEMENT_TEST(envvar)
{
  (void)pipe_file;

  std::stringstream sstr;

  sstr << "!include include_test.pipe" << std::endl;

  sprokit::pipeline_builder builder;
  builder.load_pipeline( sstr );
  sprokit::pipe_blocks const blocks = builder.pipeline_blocks();

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_declare)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_config)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_input_map)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_input_multi_map)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_output_map)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_mappings)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_all)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_missing_config_description)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_missing_input_description)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_missing_output_description)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_missing_type)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_missing_type_description)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_multiple_clusters)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_not_first)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_with_slash)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_cluster_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_input_map_with_slash)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(cluster_output_map_with_slash)
{
  sprokit::cluster_blocks const blocks = load_cluster_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 0, 1);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(process_with_slash)
{
  EXPECT_EXCEPTION(sprokit::parsing_exception,
                   load_pipe_blocks_from_file(pipe_file),
                   "with an expect error");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(connect_input_with_slash)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 1, 0);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(connect_output_with_slash)
{
  sprokit::pipe_blocks const blocks = load_pipe_blocks_from_file(pipe_file);

  test_visitor v;

  for ( auto b : blocks ) { kwiver::vital::visit( v, b ); }

  v.expect(0, 0, 1, 0);
}

// ------------------------------------------------------------------
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
::operator () (sprokit::config_pipe_block const& /*config_block*/)
{
  ++config_count;
  ++total_count;

  types.push_back(CONFIG_BLOCK);
}

void
test_visitor
::operator () (sprokit::process_pipe_block const& /*process_block*/)
{
  ++process_count;
  ++total_count;

  types.push_back(PROCESS_BLOCK);
}

void
test_visitor
::operator () (sprokit::connect_pipe_block const& /*connect_block*/)
{
  ++connect_count;
  ++total_count;

  types.push_back(CONNECT_BLOCK);
}

void
test_visitor
::operator () (sprokit::cluster_pipe_block const& /*cluster_block*/)
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

// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
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
