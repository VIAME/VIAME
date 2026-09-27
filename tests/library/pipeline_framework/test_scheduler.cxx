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
#include <viame/pipeline_framework/process.h>
#include <viame/pipeline_framework/process_factory.h>
#include <viame/pipeline_framework/scheduler.h>
#include <viame/pipeline_framework/scheduler_factory.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kv = viame;

namespace {

// ----------------------------------------------------------------------------
// Takes a number and throws, which is what a process does when a package it
// needs is broken -- the case this was found by, a cv2 that could not load.
class failing_process
  : public viame::pipeline::process
{
public:
  explicit failing_process( kv::config_block_sptr const& config )
    : process( config )
  {
    port_flags_t required;
    required.insert( flag_required );

    declare_input_port( "number", "integer", required,
                        port_description_t( "A number to fail on." ) );
    declare_output_port( "number", "integer", required,
                         port_description_t( "Never written." ) );
  }

protected:
  void _step() override
  {
    (void)grab_from_port_as< int32_t >( "number" );
    throw std::runtime_error( "failing_process: failing on purpose" );
  }
};

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
viame::pipeline::pipeline_t
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
  viame::pipeline::pipe_parser parser;
  auto const pipeline = viame::pipeline::bake_pipe_blocks(
    parser.parse_pipeline( input, "counting.pipe" ) );
  pipeline->setup_pipeline();
  return pipeline;
}

// ----------------------------------------------------------------------------
void
run_under( std::string const& type, std::string const& output )
{
  auto const pipeline = counting_pipeline( output );
  auto const scheduler = viame::pipeline::create_scheduler(
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
  EXPECT_EQ( "thread_per_process", viame::pipeline::scheduler_factory::default_type );
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
// A process that throws stops the pipeline, and wait() reports it.
//
// It used to hang instead. The thread that threw left, and the ones beside it
// were blocked on edges -- `sink` waiting for a number `fail` would never
// send -- so they never looked at the error flag and wait() joined them
// forever. `viame` did the same with a real pipeline: a cv2 that could not
// load libGL made `filter_enhance.pipe` sit until something killed it.
TEST ( scheduler, thread_per_process_stops_when_a_process_throws )
{
  scratch_output output( "scheduler_throws.txt" );

  auto const source_config = kv::config_block::empty_config();
  source_config->set_value( "start", "0" );
  source_config->set_value( "end", "1000000" );

  auto const fail_config = kv::config_block::empty_config();
  fail_config->set_value( viame::pipeline::process::config_name, "fail" );

  auto const sink_config = kv::config_block::empty_config();
  sink_config->set_value( "output", output.path() );

  auto const pipeline = std::make_shared< viame::pipeline::pipeline >();
  pipeline->add_process(
    viame::pipeline::create_process( "numbers", "source", source_config ) );
  pipeline->add_process( std::make_shared< failing_process >( fail_config ) );
  pipeline->add_process(
    viame::pipeline::create_process( "print_number", "sink", sink_config ) );
  pipeline->connect( "source", "number", "fail", "number" );
  pipeline->connect( "fail", "number", "sink", "number" );
  pipeline->setup_pipeline();

  auto const scheduler = viame::pipeline::create_scheduler(
    "thread_per_process", pipeline, kv::config_block::empty_config() );
  ASSERT_TRUE( scheduler != nullptr );

  scheduler->start();

  auto waited = std::async( std::launch::async,
                            [ &scheduler ]() { scheduler->wait(); } );

  if( waited.wait_for( std::chrono::seconds( 30 ) ) !=
      std::future_status::ready )
  {
    // A hung wait() cannot be abandoned: the future would join it on the way
    // out. Say what happened and leave.
    std::cerr << "thread_per_process_stops_when_a_process_throws: wait() was "
                 "still blocked after 30 s; the pipeline hung\n";
    std::_Exit( EXIT_FAILURE );
  }

  try
  {
    waited.get();
    FAIL() << "wait() returned normally from a pipeline whose process threw";
  }
  catch( std::runtime_error const& e )
  {
    EXPECT_STREQ( "failing_process: failing on purpose", e.what() );
  }
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
// Port frequency was here, and is gone.
//
// A port could declare that it produced or consumed more than one datum per
// step; the pipeline solved the whole graph for a consistent set of rates,
// handed each process a core frequency, and the process turned that into the
// stamp increment its output edges advanced by. The test that stood here
// recorded the producing side -- `duplicate` with `copies 2` sending each of
// three numbers three times, in a row -- and it passed, which is what made
// the removal a removal of something that worked rather than of something
// broken.
//
// Nothing in VIAME used it. The only two processes that set a frequency were
// `duplicate` and `skip`, examples of the feature, and no shipped `.pipe`
// named either. `skip` could not have worked in any case: it declared
// `1 + skip` and grabbed `skip`, so a pipeline containing it never returned
// (open question 2.13).
//
// Every port is 1:1 now, and every stamp increment is one.

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
