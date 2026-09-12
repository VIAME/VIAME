/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief What the logger writes, recorded before P8-T04 replaces it.
///
/// The log line is the only part of the logger anything outside VIAME sees:
/// DIVE reads the output of a pipeline run and picks progress and errors out
/// of it, and nothing in this tree tells us which lines it keys on. So the
/// format is the contract, down to the separators, and these tests are the
/// recording of it rather than an opinion about what it should be.
///
/// The one thing deliberately not pinned is the timestamp's value. Its
/// *shape* is pinned.

#include <viame/algorithm_framework/logger/logger.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
/// Everything written to `std::cerr` while this is alive.
///
/// The logger writes to `std::cerr` rather than to the `stderr` file
/// descriptor, so swapping its buffer is enough and the test does not have to
/// go through a temporary file.
class captured_cerr
{
public:
  captured_cerr()
    : m_saved( std::cerr.rdbuf( m_buffer.rdbuf() ) )
  {}

  ~captured_cerr()
  {
    std::cerr.rdbuf( m_saved );
  }

  std::string
  text() const
  {
    return m_buffer.str();
  }

  std::vector< std::string >
  lines() const
  {
    std::vector< std::string > out;
    std::istringstream in( m_buffer.str() );
    std::string line;

    while( std::getline( in, line ) )
    {
      out.push_back( line );
    }

    return out;
  }

private:
  std::ostringstream m_buffer;
  std::streambuf* m_saved;
};

// ----------------------------------------------------------------------------
/// A logger with a name no other test uses.
///
/// The level is read from the environment once, when the logger is first
/// created, and loggers are cached by name -- so a test that wants a
/// particular level has to ask for a name nothing has asked for yet. That is
/// itself part of the recording; see `the_level_is_fixed_when_the_logger_is_made`.
kv::logger_handle_t
logger_at( char const* level, std::string const& name )
{
  if( level )
  {
    setenv( "KWIVER_DEFAULT_LOG_LEVEL", level, 1 );
  }
  else
  {
    unsetenv( "KWIVER_DEFAULT_LOG_LEVEL" );
  }

  auto handle = kv::get_logger( name );

  unsetenv( "KWIVER_DEFAULT_LOG_LEVEL" );

  return handle;
}

// The prefix every line carries: a local timestamp to the millisecond, then
// the level, then the source location of the macro.
std::regex const prefix_re(
  R"(^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) )"
  R"((TRACE|DEBUG|INFO|WARN|ERROR|FATAL) )"
  R"(([^ ]+)\((\d+)\): (.*)$)" );

} // namespace

// ----------------------------------------------------------------------------
TEST ( logger, a_line_is_timestamp_level_location_message )
{
  auto const log = logger_at( "trace", "test.shape" );

  captured_cerr capture;
  LOG_INFO( log, "a message" );

  auto const lines = capture.lines();
  ASSERT_EQ( 1u, lines.size() );

  std::smatch match;
  ASSERT_TRUE( std::regex_match( lines[ 0 ], match, prefix_re ) )
    << "unrecognised log line: " << lines[ 0 ];

  EXPECT_EQ( "INFO", match[ 2 ] );
  EXPECT_EQ( "test_logger.cxx", match[ 3 ] );
  EXPECT_EQ( "a message", match[ 5 ] );
}

// ----------------------------------------------------------------------------
// The logger's *name* is not in the line. It selects the level and nothing
// else reaches the output, which is worth recording because it is the first
// thing someone rewriting this would assume was there.
TEST ( logger, the_logger_name_is_not_written )
{
  auto const log = logger_at( "trace", "test.name.not.written" );

  captured_cerr capture;
  LOG_INFO( log, "a message" );

  EXPECT_EQ( std::string::npos, capture.text().find( "not.written" ) );
}

// ----------------------------------------------------------------------------
TEST ( logger, every_level_has_its_own_word )
{
  auto const log = logger_at( "trace", "test.levels" );

  captured_cerr capture;
  LOG_TRACE( log, "t" );
  LOG_DEBUG( log, "d" );
  LOG_INFO( log, "i" );
  LOG_WARN( log, "w" );
  LOG_ERROR( log, "e" );

  auto const lines = capture.lines();
  ASSERT_EQ( 5u, lines.size() );

  char const* const expected[] = { "TRACE", "DEBUG", "INFO", "WARN", "ERROR" };

  for( size_t i = 0; i < 5; ++i )
  {
    std::smatch match;
    ASSERT_TRUE( std::regex_match( lines[ i ], match, prefix_re ) )
      << lines[ i ];
    EXPECT_EQ( expected[ i ], match[ 2 ] );
  }
}

// ----------------------------------------------------------------------------
// A message with newlines in it becomes several lines, each with the full
// prefix -- so a consumer reading line by line never sees a line it cannot
// parse.
TEST ( logger, a_multiline_message_repeats_the_prefix )
{
  auto const log = logger_at( "trace", "test.multiline" );

  captured_cerr capture;
  LOG_INFO( log, "first\nsecond\nthird" );

  auto const lines = capture.lines();
  ASSERT_EQ( 3u, lines.size() );

  char const* const expected[] = { "first", "second", "third" };

  for( size_t i = 0; i < 3; ++i )
  {
    std::smatch match;
    ASSERT_TRUE( std::regex_match( lines[ i ], match, prefix_re ) )
      << lines[ i ];
    EXPECT_EQ( "INFO", match[ 2 ] );
    EXPECT_EQ( expected[ i ], match[ 5 ] );
  }
}

// ----------------------------------------------------------------------------
TEST ( logger, a_level_below_the_threshold_writes_nothing )
{
  auto const log = logger_at( "warn", "test.threshold" );

  captured_cerr capture;
  LOG_TRACE( log, "trace" );
  LOG_DEBUG( log, "debug" );
  LOG_INFO( log, "info" );

  EXPECT_EQ( "", capture.text() );

  captured_cerr kept;
  LOG_WARN( log, "warn" );
  LOG_ERROR( log, "error" );

  EXPECT_EQ( 2u, kept.lines().size() );
}

// ----------------------------------------------------------------------------
TEST ( logger, the_enabled_predicates_agree_with_the_threshold )
{
  auto const log = logger_at( "info", "test.predicates" );

  EXPECT_FALSE( IS_TRACE_ENABLED( log ) );
  EXPECT_FALSE( IS_DEBUG_ENABLED( log ) );
  EXPECT_TRUE( IS_INFO_ENABLED( log ) );
  EXPECT_TRUE( IS_WARN_ENABLED( log ) );
  EXPECT_TRUE( IS_ERROR_ENABLED( log ) );
  EXPECT_TRUE( IS_FATAL_ENABLED( log ) );
}

// ----------------------------------------------------------------------------
// An unrecognised level is ignored rather than rejected: the logger keeps the
// level it would have had.
TEST ( logger, an_unknown_level_name_is_ignored )
{
  auto const log = logger_at( "verbose", "test.unknown.level" );

  EXPECT_TRUE( IS_ERROR_ENABLED( log ) );
}

// ----------------------------------------------------------------------------
// The level names are matched case-insensitively.
TEST ( logger, the_level_name_is_not_case_sensitive )
{
  auto const log = logger_at( "ERROR", "test.upper.level" );

  EXPECT_FALSE( IS_WARN_ENABLED( log ) );
  EXPECT_TRUE( IS_ERROR_ENABLED( log ) );
}

// ----------------------------------------------------------------------------
// Asking twice for the same name gives the same logger, which is what makes
// it worth caching the handle in a member.
TEST ( logger, the_same_name_is_the_same_logger )
{
  EXPECT_EQ( kv::get_logger( "test.identity" ),
    kv::get_logger( "test.identity" ) );
}

// ----------------------------------------------------------------------------
// And because it is the same logger, the environment is read once. A process
// that changes the variable after its first log call changes nothing.
TEST ( logger, the_level_is_fixed_when_the_logger_is_made )
{
  auto const log = logger_at( "error", "test.fixed.level" );
  ASSERT_FALSE( IS_WARN_ENABLED( log ) );

  auto const again = logger_at( "trace", "test.fixed.level" );
  EXPECT_FALSE( IS_WARN_ENABLED( again ) );
}

// ----------------------------------------------------------------------------
// `set_level` is how a program changes it deliberately.
TEST ( logger, set_level_changes_the_threshold )
{
  auto const log = logger_at( "error", "test.set.level" );
  ASSERT_FALSE( IS_INFO_ENABLED( log ) );

  log->set_level( kv::kwiver_logger::LEVEL_INFO );

  EXPECT_TRUE( IS_INFO_ENABLED( log ) );
  EXPECT_EQ( kv::kwiver_logger::LEVEL_INFO, log->get_level() );

  log->set_level( kv::kwiver_logger::LEVEL_ERROR );
}

// ----------------------------------------------------------------------------
TEST ( logger, log_assert_writes_only_when_the_condition_fails )
{
  auto const log = logger_at( "trace", "test.assert" );

  {
    captured_cerr capture;
    LOG_ASSERT( log, 1 == 1, "not reached" );
    EXPECT_EQ( "", capture.text() );
  }

  captured_cerr capture;
  LOG_ASSERT( log, 1 == 2, "explanation" );

  auto const lines = capture.lines();
  ASSERT_EQ( 2u, lines.size() );

  // Two lines, because the message the macro builds has a newline in it: the
  // condition it was given, then whatever the caller wanted to say about it.
  std::smatch first;
  ASSERT_TRUE( std::regex_match( lines[ 0 ], first, prefix_re ) ) << lines[ 0 ];
  EXPECT_EQ( "ERROR", first[ 2 ] );
  EXPECT_EQ( "ASSERTION FAILED: (1 == 2)", first[ 5 ] );

  std::smatch second;
  ASSERT_TRUE( std::regex_match( lines[ 1 ], second, prefix_re ) )
    << lines[ 1 ];
  EXPECT_EQ( "ERROR", second[ 2 ] );
  EXPECT_EQ( "explanation", second[ 5 ] );
}

// ----------------------------------------------------------------------------
// A callback sees the message, the logger's name and the source location --
// the parts the formatted line drops. `plugin_explorer` and the pipeline
// runner are what this exists for.
TEST ( logger, a_global_callback_sees_what_the_line_does_not )
{
  auto const log = logger_at( "trace", "test.callback" );

  kv::kwiver_logger::log_level_t seen_level{};
  std::string seen_name;
  std::string seen_message;
  std::string seen_file;

  kv::kwiver_logger::set_global_callback(
    [ & ]( kv::kwiver_logger::log_level_t level, std::string const& name,
           std::string const& msg,
           kwiver::vital::logger_ns::location_info const& loc )
    {
      seen_level = level;
      seen_name = name;
      seen_message = msg;
      seen_file = loc.get_file_name();
    } );

  {
    captured_cerr capture;
    LOG_WARN( log, "through the callback" );
  }

  kv::kwiver_logger::set_global_callback( nullptr );

  EXPECT_EQ( kv::kwiver_logger::LEVEL_WARN, seen_level );
  EXPECT_EQ( "test.callback", seen_name );
  EXPECT_EQ( "through the callback", seen_message );
  EXPECT_EQ( "test_logger.cxx", seen_file );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
