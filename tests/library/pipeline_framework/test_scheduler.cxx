/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The C++ schedulers, recorded before P8-T07 removes one of them.
///
/// Every shipped pipeline names `pythread_per_process`, so the two C++
/// schedulers are reached only as the default -- `thread_per_process`, which
/// `scheduler_factory::default_type` names and which anything embedding a
/// pipeline gets -- and, for `sync`, not at all.
///
/// That is the argument for removing `sync` and it is also the reason the
/// two need a test of their own: 292 pipelines going green says nothing
/// about either. What is recorded here is that a pipeline baked from text
/// runs to completion under each, and produces the same output through both,
/// which is the only property a scheduler has that a caller can see.

#include <viame/pipeline_framework/pipe_bakery.h>
#include <viame/pipeline_framework/pipe_parser.h>
#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/scheduler.h>
#include <viame/pipeline_framework/scheduler_factory.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
class scratch_output
{
public:
  explicit scratch_output( std::string const& name ) : m_path( name )
  {
    std::remove( m_path.c_str() );
  }

  ~scratch_output() { std::remove( m_path.c_str() ); }

  std::string const& path() const { return m_path; }

  std::vector< std::string > lines() const
  {
    std::vector< std::string > out;
    std::ifstream in( m_path );
    for( std::string line; std::getline( in, line ); )
    {
      out.push_back( line );
    }
    return out;
  }

private:
  std::string m_path;
};

// ----------------------------------------------------------------------------
// Five numbers into a file, which is the smallest pipeline that has an
// upstream, a downstream, an edge between them and an observable result.
sprokit::pipeline_t
counting_pipeline( std::string const& output )
{
  std::ostringstream text;
  text << "process source\n"
       << "  :: numbers\n"
       << "  :start 0\n"
       << "  :end   5\n"
       << "process sink\n"
       << "  :: print_number\n"
       << "  :output " << output << "\n"
       << "connect from source.number\n"
       << "        to   sink.number\n";

  std::istringstream input( text.str() );
  sprokit::pipe_parser parser;
  auto const pipeline = sprokit::bake_pipe_blocks(
    parser.parse_pipeline( input, "counting.pipe" ) );
  pipeline->setup_pipeline();
  return pipeline;
}

// ----------------------------------------------------------------------------
void
run_under( std::string const& type, std::string const& output )
{
  auto const pipeline = counting_pipeline( output );
  auto const scheduler = sprokit::create_scheduler(
    type, pipeline, kv::config_block::empty_config() );
  ASSERT_TRUE( scheduler != nullptr ) << "no scheduler named " << type;

  scheduler->start();
  scheduler->wait();
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( scheduler, the_default_is_thread_per_process )
{
  // Named rather than asserted about behaviour: this is what a pipeline that
  // says nothing about scheduling gets, and every embedded pipeline says
  // nothing.
  EXPECT_EQ( "thread_per_process", sprokit::scheduler_factory::default_type );
}

// ----------------------------------------------------------------------------
TEST ( scheduler, thread_per_process_runs_a_pipeline_to_completion )
{
  scratch_output output( "scheduler_tpp.txt" );
  run_under( "thread_per_process", output.path() );

  EXPECT_EQ( ( std::vector< std::string >{ "0", "1", "2", "3", "4" } ),
             output.lines() );
}

// ----------------------------------------------------------------------------
// `end` is exclusive, which the pipeline above depends on and which nothing
// else in the tree writes down.
TEST ( scheduler, the_range_excludes_its_end )
{
  scratch_output output( "scheduler_range.txt" );
  run_under( "thread_per_process", output.path() );

  ASSERT_FALSE( output.lines().empty() );
  EXPECT_EQ( "4", output.lines().back() );
}

// ----------------------------------------------------------------------------
// The two C++ schedulers differ in how they step the pipeline and in nothing
// a caller can observe. This is what makes `sync` removable rather than
// merely unused: it is not a different answer, it is the same answer
// arrived at on one thread.
TEST ( scheduler, sync_and_thread_per_process_agree )
{
  scratch_output threaded( "scheduler_threaded.txt" );
  scratch_output synchronous( "scheduler_sync.txt" );

  run_under( "thread_per_process", threaded.path() );
  run_under( "sync", synchronous.path() );

  EXPECT_EQ( threaded.lines(), synchronous.lines() );
  EXPECT_EQ( ( std::vector< std::string >{ "0", "1", "2", "3", "4" } ),
             synchronous.lines() );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );

  // The example processes and the schedulers are plugins; without this the
  // bake finds no `numbers` and the factory no `thread_per_process`.
  kv::plugin_manager::instance().load_all_plugins();

  return RUN_ALL_TESTS();
}
