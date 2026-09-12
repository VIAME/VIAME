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
/// That was the argument for removing `sync` -- P8-T07 did -- and it is also
/// the reason `thread_per_process` needs a test of its own: 292 pipelines
/// going green says nothing about it. What is recorded here is that a
/// pipeline baked from text runs to completion under it and produces the
/// numbers it should, which is the only property a scheduler has that a
/// caller can see.

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
// `sync` was here. It ran the same pipeline on one thread and produced
// byte-identical output -- the test asserted exactly that, and passed, which
// is what made it removable rather than merely unused: not a different
// answer, the same answer arrived at differently.
//
// P8-T07 removed it, so the test went with it. The measurement it made is
// the reason recorded in `tests/baseline/removed.json`, and this is where it
// was made.

// ----------------------------------------------------------------------------
// Port frequency, recorded before P8-T07 removes it.
//
// A port may declare that it produces or consumes more than one datum per
// step. `pipeline::check_port_frequencies` solves the whole graph for a
// consistent set of rates, hands each process a core frequency, and the
// process turns that into the stamp increment its output edges advance by.
// Getting it wrong does not fail to compile and does not throw: it drops or
// duplicates data, which looks like a bad pipeline rather than a bad
// framework.
//
// **Nothing in VIAME uses it.** The only two processes that call
// `set_output_port_frequency` or `set_input_port_frequency` are `duplicate`
// and `skip`, which are examples demonstrating the feature, and no shipped
// `.pipe` names either. That is the argument for removing it -- and the
// reason it needs a test first, because the 292 pipelines that do run are
// all 1:1 and would not notice the machinery going wrong.
TEST ( scheduler, a_port_may_produce_more_than_one_datum_per_step )
{
  scratch_output output( "scheduler_duplicate.txt" );

  std::ostringstream text;
  text << "process source\n"
       << "  :: numbers\n"
       << "  :start 0\n"
       << "  :end   3\n"
       << "process dup\n"
       << "  :: duplicate\n"
       << "  :copies 2\n"
       << "process sink\n"
       << "  :: print_number\n"
       << "  :output " << output.path() << "\n"
       << "connect from source.number\n"
       << "        to   dup.input\n"
       << "connect from dup.duplicate\n"
       << "        to   sink.number\n";

  std::istringstream input( text.str() );
  sprokit::pipe_parser parser;
  auto const pipeline = sprokit::bake_pipe_blocks(
    parser.parse_pipeline( input, "duplicate.pipe" ) );
  pipeline->setup_pipeline();

  auto const scheduler = sprokit::create_scheduler(
    "thread_per_process", pipeline, kv::config_block::empty_config() );
  ASSERT_TRUE( scheduler != nullptr );
  scheduler->start();
  scheduler->wait();

  // `copies 2` means the port frequency is 1 + 2 = 3, so each of the three
  // input numbers reaches the sink three times, in a row.
  EXPECT_EQ( ( std::vector< std::string >{ "0", "0", "0",
                                           "1", "1", "1",
                                           "2", "2", "2" } ),
             output.lines() );
}

// ----------------------------------------------------------------------------
// The other direction -- a port that *consumes* several data per step -- has
// no test here, because the only process that declares one **deadlocks**.
//
// `skip_process` declares `set_input_port_frequency( input, 1 + skip )` and
// its `_step` then grabs `skip` data, one fewer than it said it would. The
// stamp bookkeeping expects the declared rate, so a pipeline containing it
// stops: no error, no output, just a scheduler that never returns. It is the
// only `skip` there is and no shipped `.pipe` names it, so nothing has ever
// run it.
//
// Left as it is. P8-T07 is removing port frequency, and a one-character fix
// to an example of the feature being removed would be work spent in the
// wrong direction; what matters is that the deadlock is written down, in the
// `removed.json` entry and in open question 2.13, rather than being
// rediscovered by whoever next wonders what `skip` was for.

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
