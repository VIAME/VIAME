/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Process type aliases, added by P8-T07.
///
/// A renamed process used to be two entries in the compatibility baseline: a
/// removal and an addition, indistinguishable from a process that was
/// actually deleted and an unrelated one that appeared. Algorithms have had
/// aliases since the imports; processes could not, because
/// `register_process` takes the name from a static on the class, so there
/// was nowhere to put a second one.
///
/// Resolution is in `create_process` rather than in the bakery itself, which
/// covers every route into a process -- the pipe bakery, the embedded
/// pipeline, and anything else that names a type -- rather than only the one
/// the task text named.

#include <viame/pipeline_framework/pipe_bakery.h>
#include <viame/pipeline_framework/pipe_parser.h>
#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/process_registry_exception.h>
#include <viame/pipeline_framework/process_factory.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>

#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
sprokit::pipeline_t
bake( std::string const& text )
{
  std::istringstream input( text );
  sprokit::pipe_parser parser;
  return sprokit::bake_pipe_blocks(
    parser.parse_pipeline( input, "alias.pipe" ) );
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( process_alias, an_unregistered_type_is_still_an_error )
{
  // The table is consulted only after the real name has failed, so a type
  // that is neither registered nor aliased fails exactly as before.
  EXPECT_THROW(
    sprokit::create_process( "not_a_process", "p",
                             kv::config_block::empty_config() ),
    sprokit::no_such_process_type_exception );
}

// ----------------------------------------------------------------------------
TEST ( process_alias, an_alias_resolves_to_its_target )
{
  sprokit::add_process_alias( "old_numbers", "numbers" );

  auto const proc = sprokit::create_process(
    "old_numbers", "source", kv::config_block::empty_config() );

  ASSERT_TRUE( proc != nullptr );
  EXPECT_EQ( "source", proc->name() );

  // The process is told the name it actually is, not the one the caller
  // used: `viame pipe-check` reports what a pipeline resolves to, and an
  // alias that reported itself would make a rename invisible in the other
  // direction.
  EXPECT_EQ( "numbers", proc->type() );
}

// ----------------------------------------------------------------------------
TEST ( process_alias, a_pipeline_may_name_the_old_type )
{
  sprokit::add_process_alias( "old_print_number", "print_number" );

  auto const pipeline = bake(
    "process source\n"
    "  :: numbers\n"
    "  :start 0\n"
    "  :end   2\n"
    "process sink\n"
    "  :: old_print_number\n"
    "  :output alias_out.txt\n"
    "connect from source.number\n"
    "        to   sink.number\n" );

  ASSERT_TRUE( pipeline != nullptr );
  EXPECT_EQ( "print_number", pipeline->process_by_name( "sink" )->type() );
}

// ----------------------------------------------------------------------------
// An alias whose target is not registered is a mistake in whoever added the
// alias, not in the pipeline that used it, so the error names the type the
// caller wrote rather than the one it resolved to.
TEST ( process_alias, an_alias_to_nothing_reports_the_name_that_was_used )
{
  sprokit::add_process_alias( "points_nowhere", "also_not_a_process" );

  try
  {
    sprokit::create_process( "points_nowhere", "p",
                             kv::config_block::empty_config() );
    FAIL() << "expected no_such_process_type_exception";
  }
  catch( sprokit::no_such_process_type_exception const& e )
  {
    EXPECT_NE( std::string::npos,
               std::string( e.what() ).find( "points_nowhere" ) );
  }
}

// ----------------------------------------------------------------------------
TEST ( process_alias, the_table_is_readable )
{
  sprokit::add_process_alias( "readable_alias", "numbers" );

  auto const aliases = sprokit::process_aliases();
  auto const entry = aliases.find( "readable_alias" );

  ASSERT_NE( aliases.end(), entry );
  EXPECT_EQ( "numbers", entry->second );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  kv::plugin_manager::instance().load_all_plugins();
  return RUN_ALL_TESTS();
}
