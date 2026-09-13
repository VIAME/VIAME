/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief What `config_block` means, recorded before P8-T09 simplifies it.
///
/// Every pipeline, every algorithm and every applet in VIAME reads its
/// parameters through this class, and nothing in the tree wrote down what it
/// does. P8-T09's instruction is "keep `config_block` semantics" while
/// removing what nothing uses -- which requires knowing which of its
/// behaviours are semantics and which are accidents.
///
/// The answers below were taken by running it. Several are surprising and
/// are recorded rather than corrected:
///
///  * `subblock` copies and `subblock_view` does not, so writing through one
///    reaches the parent and writing through the other does not;
///  * a read-only key is not an error to *set*, it is silently refused --
///    no, it throws, and which of those it is matters to every caller that
///    sets a default;
///  * `get_value` with a default swallows every failure, including a value
///    that is present but unconvertible;
///  * `merge_config` overwrites, and the direction is the opposite of what
///    the name suggests to about half of readers.

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/config_block_exception.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace kv = kwiver::vital;

namespace {

kv::config_block_sptr
a_config()
{
  auto config = kv::config_block::empty_config();
  config->set_value( "top", "1" );
  config->set_value( "block:inner", "2" );
  config->set_value( "block:deeper:leaf", "3" );
  return config;
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( config_block, the_separator_is_a_colon )
{
  EXPECT_EQ( ":", kv::config_block::block_sep() );
}

// ----------------------------------------------------------------------------
TEST ( config_block, values_are_strings_and_convert_on_the_way_out )
{
  auto const config = kv::config_block::empty_config();
  config->set_value( "count", 42 );
  config->set_value( "ratio", 0.5 );
  config->set_value( "flag", true );
  config->set_value( "name", "fish" );

  EXPECT_EQ( 42, config->get_value< int >( "count" ) );
  EXPECT_DOUBLE_EQ( 0.5, config->get_value< double >( "ratio" ) );
  EXPECT_TRUE( config->get_value< bool >( "flag" ) );
  EXPECT_EQ( "fish", config->get_value< std::string >( "name" ) );

  // Everything is stored as text, so the int reads back as one.
  EXPECT_EQ( "42", config->get_value< std::string >( "count" ) );

  // A `bool` is stored as `"1"`, not as the word. So a config written by
  // C++ and one written by hand in a `.pipe` do not look the same, and the
  // reader has to accept both -- which it does, below.
  EXPECT_EQ( "1", config->get_value< std::string >( "flag" ) );
}

// ----------------------------------------------------------------------------
// The spellings `bool` accepts. A pipeline writer types one of these and a
// reader has to keep meaning it. `on` and `off` are **not** among them,
// which is worth knowing before anyone "tidies" the list: they read as an
// obvious pair with the others and would throw.
TEST ( config_block, the_words_that_mean_true )
{
  auto const config = kv::config_block::empty_config();

  for( auto const& word : { "true", "yes", "1", "TRUE", "Yes" } )
  {
    config->set_value( "flag", word );
    EXPECT_TRUE( config->get_value< bool >( "flag" ) ) << word;
  }

  for( auto const& word : { "false", "no", "0", "FALSE", "No" } )
  {
    config->set_value( "flag", word );
    EXPECT_FALSE( config->get_value< bool >( "flag" ) ) << word;
  }

  for( auto const& word : { "on", "off", "y", "n", "" } )
  {
    config->set_value( "flag", word );
    EXPECT_THROW( config->get_value< bool >( "flag" ),
                  kv::bad_config_block_cast_exception ) << word;
  }
}

// ----------------------------------------------------------------------------
TEST ( config_block, a_missing_key_throws )
{
  auto const config = a_config();

  EXPECT_THROW( config->get_value< int >( "absent" ),
                kv::no_such_configuration_value_exception );
  EXPECT_FALSE( config->has_value( "absent" ) );
  EXPECT_TRUE( config->has_value( "top" ) );
}

// ----------------------------------------------------------------------------
// The defaulted overload is `noexcept` and swallows everything: a key that is
// absent, and a key that is present but holds something the conversion
// cannot read. The second is the one that surprises -- a typo in a pipeline
// gives you the default rather than an error.
TEST ( config_block, a_default_swallows_a_bad_value_too )
{
  auto const config = kv::config_block::empty_config();
  config->set_value( "number", "not a number" );

  EXPECT_EQ( 7, config->get_value< int >( "absent", 7 ) );
  EXPECT_EQ( 7, config->get_value< int >( "number", 7 ) );

  // Without the default it is an error, and a different one from "missing":
  // `bad_config_block_cast_exception`, which the conversion's own
  // `bad_config_block_cast` is upgraded into so that the message can name
  // the key.
  EXPECT_THROW( config->get_value< int >( "number" ),
                kv::bad_config_block_cast_exception );
}

// ----------------------------------------------------------------------------
// `subblock` copies. Writing to it does not reach the config it came from.
TEST ( config_block, subblock_is_a_copy )
{
  auto const config = a_config();
  auto const block = config->subblock( "block" );

  EXPECT_EQ( "2", block->get_value< std::string >( "inner" ) );
  EXPECT_EQ( "3", block->get_value< std::string >( "deeper:leaf" ) );
  EXPECT_FALSE( block->has_value( "top" ) );

  block->set_value( "inner", "changed" );
  EXPECT_EQ( "2", config->get_value< std::string >( "block:inner" ) );
}

// ----------------------------------------------------------------------------
// `subblock_view` does not copy. Writing to it reaches the parent, and a
// value added to the parent afterwards appears in the view.
TEST ( config_block, subblock_view_is_a_window )
{
  auto const config = a_config();
  auto const view = config->subblock_view( "block" );

  EXPECT_EQ( "2", view->get_value< std::string >( "inner" ) );

  view->set_value( "inner", "changed" );
  EXPECT_EQ( "changed", config->get_value< std::string >( "block:inner" ) );

  config->set_value( "block:added", "later" );
  EXPECT_EQ( "later", view->get_value< std::string >( "added" ) );
}

// ----------------------------------------------------------------------------
TEST ( config_block, read_only_keys_refuse_to_change )
{
  auto const config = kv::config_block::empty_config();
  config->set_value( "fixed", "original" );
  config->mark_read_only( "fixed" );

  EXPECT_TRUE( config->is_read_only( "fixed" ) );
  EXPECT_THROW( config->set_value( "fixed", "other" ),
                kv::set_on_read_only_value_exception );
  EXPECT_EQ( "original", config->get_value< std::string >( "fixed" ) );

  EXPECT_THROW( config->unset_value( "fixed" ),
                kv::unset_on_read_only_value_exception );
}

// ----------------------------------------------------------------------------
// Which way round `merge_config` goes: the argument wins.
TEST ( config_block, merge_lets_the_argument_win )
{
  auto const mine = kv::config_block::empty_config();
  mine->set_value( "shared", "mine" );
  mine->set_value( "only_mine", "kept" );

  auto const theirs = kv::config_block::empty_config();
  theirs->set_value( "shared", "theirs" );
  theirs->set_value( "only_theirs", "added" );

  mine->merge_config( theirs );

  EXPECT_EQ( "theirs", mine->get_value< std::string >( "shared" ) );
  EXPECT_EQ( "kept", mine->get_value< std::string >( "only_mine" ) );
  EXPECT_EQ( "added", mine->get_value< std::string >( "only_theirs" ) );
}

// ----------------------------------------------------------------------------
TEST ( config_block, available_values_are_whole_paths )
{
  auto const config = a_config();
  auto values = config->available_values();
  std::sort( values.begin(), values.end() );

  EXPECT_EQ( ( kv::config_block_keys_t{ "block:deeper:leaf",
                                        "block:inner",
                                        "top" } ),
             values );
}

// ----------------------------------------------------------------------------
// A description travels with the value and survives being set again only if
// the second set supplies one -- which is why a `PARAM_DEFAULT` that is
// later overridden by a pipeline keeps its help text.
TEST ( config_block, descriptions_are_kept_unless_replaced )
{
  auto const config = kv::config_block::empty_config();
  config->set_value( "key", "one", "what it means" );
  EXPECT_EQ( "what it means", config->get_description( "key" ) );

  config->set_value( "key", "two" );
  EXPECT_EQ( "what it means", config->get_description( "key" ) );
  EXPECT_EQ( "two", config->get_value< std::string >( "key" ) );

  config->set_value( "key", "three", "something else" );
  EXPECT_EQ( "something else", config->get_description( "key" ) );
}

// ----------------------------------------------------------------------------
TEST ( config_block, a_value_can_be_read_as_a_vector )
{
  auto const config = kv::config_block::empty_config();
  config->set_value( "sizes", "1 2 3" );

  auto const sizes = config->get_value_as_vector< int >( "sizes", " " );
  EXPECT_EQ( ( std::vector< int >{ 1, 2, 3 } ), sizes );
}

// ----------------------------------------------------------------------------
TEST ( config_block, unsetting_removes_the_key_entirely )
{
  auto const config = a_config();
  EXPECT_TRUE( config->has_value( "top" ) );

  config->unset_value( "top" );
  EXPECT_FALSE( config->has_value( "top" ) );
  EXPECT_THROW( config->get_value< int >( "top" ),
                kv::no_such_configuration_value_exception );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
