/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The `.pipe` grammar, recorded before P8-T07 trims the library that
/// parses it.
///
/// `tests/baseline/pipes.json` already says that all 292 shipped pipelines
/// bake and resolve, which is a strong thing to have. What it cannot say is
/// what the parser does with anything those 292 happen not to contain -- and
/// P8-T07 removes whole grammar productions, so "no shipped pipeline uses
/// it" is the argument for removal rather than the check on it.
///
/// So this is the grammar itself, block by block, taken by running the
/// parser. Where a rule looks surprising it is recorded rather than
/// corrected: `:=` and `=` mean different things, a flag list is attached to
/// the key and not the value, and a `config` block's name becomes a prefix
/// on every key inside it.
///
/// Kwiver shipped python tests for this -- `test-load.py` and
/// `test-bake.py` -- and they have never run on this branch: they are gated
/// behind `KWIVER_ENABLE_PYTHON_TESTS`, which also gates three other test
/// trees whose fixtures earlier phases pruned, so turning it on does not
/// configure. See open question 2.12.

#include <viame/pipeline_framework/pipe_parser.h>
#include <viame/pipeline_framework/pipe_declaration_types.h>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
sprokit::pipe_blocks
parse( std::string const& text )
{
  std::istringstream input( text );
  sprokit::pipe_parser parser;
  return parser.parse_pipeline( input, "test.pipe" );
}

// ----------------------------------------------------------------------------
// The blocks come back as a variant, and every test below wants one of the
// three kinds by index rather than all of them in order.
template < typename Block >
std::vector< Block >
only( sprokit::pipe_blocks const& blocks )
{
  std::vector< Block > out;
  for( auto const& block : blocks )
  {
    if( auto const* const wanted = std::get_if< Block >( &block ) )
    {
      out.push_back( *wanted );
    }
  }
  return out;
}

using configs = std::vector< sprokit::config_pipe_block >;
using processes = std::vector< sprokit::process_pipe_block >;
using connections = std::vector< sprokit::connect_pipe_block >;

// ----------------------------------------------------------------------------
std::string
joined( kwiver::vital::config_block_keys_t const& keys )
{
  std::string out;
  for( auto const& key : keys )
  {
    if( !out.empty() ) { out += ':'; }
    out += key;
  }
  return out;
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( pipe_parser, a_process_block_is_a_name_and_a_type )
{
  auto const blocks = parse(
    "process reader\n"
    "  :: frame_list_input\n" );

  auto const found = only< sprokit::process_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  EXPECT_EQ( "reader", found[ 0 ].name );
  EXPECT_EQ( "frame_list_input", found[ 0 ].type );
  EXPECT_TRUE( found[ 0 ].config_values.empty() );
}

// ----------------------------------------------------------------------------
// The keys that follow a process block belong to it, and the block's own
// name is not part of them: the process is the context, so `:image_list_file`
// is one key and not two.
TEST ( pipe_parser, a_process_keeps_its_own_config )
{
  auto const blocks = parse(
    "process reader\n"
    "  :: frame_list_input\n"
    "  :image_list_file  input.txt\n"
    "  :frame_time       0.03333\n" );

  auto const found = only< sprokit::process_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  ASSERT_EQ( 2u, found[ 0 ].config_values.size() );

  EXPECT_EQ( "image_list_file", joined( found[ 0 ].config_values[ 0 ].key_path ) );
  EXPECT_EQ( "input.txt", found[ 0 ].config_values[ 0 ].value );
  EXPECT_EQ( "frame_time", joined( found[ 0 ].config_values[ 1 ].key_path ) );
  EXPECT_EQ( "0.03333", found[ 0 ].config_values[ 1 ].value );
}

// ----------------------------------------------------------------------------
// A value runs to the end of the line, spaces and all, and is not quoted,
// unescaped or trimmed on the right of its first word. This is why a path
// with a space in it works and a trailing comment does not.
TEST ( pipe_parser, a_value_is_the_rest_of_the_line )
{
  auto const blocks = parse(
    "process reader\n"
    "  :: frame_list_input\n"
    "  :image_list_file  a path with spaces.txt\n" );

  auto const found = only< sprokit::process_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  ASSERT_EQ( 1u, found[ 0 ].config_values.size() );
  EXPECT_EQ( "a path with spaces.txt", found[ 0 ].config_values[ 0 ].value );
}

// ----------------------------------------------------------------------------
TEST ( pipe_parser, a_config_block_prefixes_every_key_in_it )
{
  auto const blocks = parse(
    "config detector\n"
    "  :type          netharn\n"
    "  :netharn:mode  detector\n" );

  auto const found = only< sprokit::config_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  EXPECT_EQ( "detector", joined( found[ 0 ].key ) );
  ASSERT_EQ( 2u, found[ 0 ].values.size() );

  // The block's name is *not* folded into the key here: the block carries it
  // and the values carry the rest. Anything reading these blocks has to put
  // the two together itself.
  EXPECT_EQ( "type", joined( found[ 0 ].values[ 0 ].key_path ) );
  EXPECT_EQ( "netharn:mode", joined( found[ 0 ].values[ 1 ].key_path ) );
}

// ----------------------------------------------------------------------------
TEST ( pipe_parser, a_connection_is_two_addresses )
{
  auto const blocks = parse(
    "connect from reader.image\n"
    "        to   detector.image\n" );

  auto const found = only< sprokit::connect_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  EXPECT_EQ( "reader", found[ 0 ].from.first );
  EXPECT_EQ( "image", found[ 0 ].from.second );
  EXPECT_EQ( "detector", found[ 0 ].to.first );
  EXPECT_EQ( "image", found[ 0 ].to.second );
}

// ----------------------------------------------------------------------------
// The flags sit on the key, before the value, and come back as a list on the
// entry rather than being interpreted here.
TEST ( pipe_parser, flags_belong_to_the_key )
{
  auto const blocks = parse(
    "config global\n"
    "  :fixed[RO]  1\n"
    "  :plain      3\n" );

  auto const found = only< sprokit::config_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  ASSERT_EQ( 2u, found[ 0 ].values.size() );

  EXPECT_EQ( "fixed", joined( found[ 0 ].values[ 0 ].key_path ) );
  ASSERT_EQ( 1u, found[ 0 ].values[ 0 ].flags.size() );
  EXPECT_EQ( "RO", found[ 0 ].values[ 0 ].flags[ 0 ] );
  EXPECT_EQ( "1", found[ 0 ].values[ 0 ].value );

  EXPECT_TRUE( found[ 0 ].values[ 1 ].flags.empty() );
}

// ----------------------------------------------------------------------------
// A list of two flags, which the grammar in `parse_attrs` has always claimed
// and which the function could not parse until P8-T07:
//
//     attr-list ::= attr
//                 | attr ',' attr_list
//
// It accepted the comma and then went round the loop without fetching the
// token after it, so the comma was itself tested for being a flag name.
// No shipped pipeline writes two flags, which is what one would expect of a
// production that never worked.
TEST ( pipe_parser, a_bracket_may_hold_more_than_one_flag )
{
  auto const blocks = parse(
    "config global\n"
    "  :one[ro]              1\n"
    "  :two[ro,local]        2\n"
    "  :three[ro,local,tunable] 3\n" );

  auto const found = only< sprokit::config_pipe_block >( blocks );
  ASSERT_EQ( 1u, found.size() );
  ASSERT_EQ( 3u, found[ 0 ].values.size() );

  EXPECT_EQ( ( std::vector< std::string >{ "ro" } ),
             found[ 0 ].values[ 0 ].flags );
  EXPECT_EQ( ( std::vector< std::string >{ "ro", "local" } ),
             found[ 0 ].values[ 1 ].flags );
  EXPECT_EQ( ( std::vector< std::string >{ "ro", "local", "tunable" } ),
             found[ 0 ].values[ 2 ].flags );

  // The value is still the rest of the line after the bracket.
  EXPECT_EQ( "2", found[ 0 ].values[ 1 ].value );
}

// ----------------------------------------------------------------------------
// A comment runs from `#` to the end of the line, and a line that is only a
// comment produces no block at all.
TEST ( pipe_parser, comments_produce_nothing )
{
  auto const blocks = parse(
    "# a whole line\n"
    "\n"
    "process reader\n"
    "  :: frame_list_input\n"
    "# another\n" );

  EXPECT_EQ( 1u, blocks.size() );
  EXPECT_EQ( 1u, only< sprokit::process_pipe_block >( blocks ).size() );
}

// ----------------------------------------------------------------------------
// Every block records where it came from, which is the whole reason a
// pipeline error can name a file and a line.
TEST ( pipe_parser, every_block_remembers_its_line )
{
  auto const blocks = parse(
    "\n"
    "\n"
    "process reader\n"
    "  :: frame_list_input\n"
    "\n"
    "connect from reader.image\n"
    "        to   detector.image\n" );

  auto const p = only< sprokit::process_pipe_block >( blocks );
  auto const c = only< sprokit::connect_pipe_block >( blocks );
  ASSERT_EQ( 1u, p.size() );
  ASSERT_EQ( 1u, c.size() );

  EXPECT_EQ( "test.pipe", p[ 0 ].loc.file() );
  EXPECT_EQ( 3, p[ 0 ].loc.line() );
  EXPECT_EQ( 6, c[ 0 ].loc.line() );
}

// ----------------------------------------------------------------------------
// Blocks come back in the order they were written, and a `config` after a
// `process` is a separate block rather than more of the process.
TEST ( pipe_parser, order_is_preserved_and_blocks_do_not_merge )
{
  auto const blocks = parse(
    "process a\n"
    "  :: one\n"
    "config global\n"
    "  :key value\n"
    "process b\n"
    "  :: two\n" );

  ASSERT_EQ( 3u, blocks.size() );
  EXPECT_TRUE( std::holds_alternative< sprokit::process_pipe_block >( blocks[ 0 ] ) );
  EXPECT_TRUE( std::holds_alternative< sprokit::config_pipe_block >( blocks[ 1 ] ) );
  EXPECT_TRUE( std::holds_alternative< sprokit::process_pipe_block >( blocks[ 2 ] ) );

  auto const found = only< sprokit::process_pipe_block >( blocks );
  EXPECT_EQ( "a", found[ 0 ].name );
  EXPECT_EQ( "b", found[ 1 ].name );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
