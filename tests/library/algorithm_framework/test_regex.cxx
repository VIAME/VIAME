/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The four patterns VIAME's parsers use, recorded before P8-T05
/// changed the engine under them.
///
/// `kwiversys::RegularExpression` was a POSIX-style engine and `std::regex`
/// defaults to ECMAScript, so the question was whether these four patterns
/// mean the same thing to both. The answers below were taken by running the
/// old engine; they agree with the new one, which is what makes the swap a
/// change of implementation rather than of behaviour.
///
/// Three of the four read configuration a user wrote, so a difference here
/// would show up as a pipeline that stopped parsing.

#include <viame/algorithm_framework/util/regex.h>

#include <gtest/gtest.h>

#include <string>

namespace kv = viame;

namespace {

// The pipeline tokeniser's idea of a left-hand-side word: starts with a
// letter, is at least three characters, and does not end in `:`.
constexpr char const* word_pattern =
  "^[a-zA-Z][-a-zA-Z0-9.:/_]+[-a-zA-Z0-9./_]";

// The `[RO]` and `[ro,local]` flags a config key can carry.
constexpr char const* flag_pattern = "^\\[[a-zA-Z,]+\\]";

// `$ENV{HOME}`, and the `$CONFIG{a.b}` the token expander fills in.
constexpr char const* token_pattern =
  "\\$([a-zA-Z][a-zA-Z0-9_]*)\\{([a-zA-Z0-9._:]+)?\\}";

// `$person/3.5/0 0 255` -- a class name, a line thickness, and a colour.
constexpr char const* colour_pattern =
  "\\$([^/]+)/([0-9.]+)/([0-9]+) ([0-9]+) ([0-9]+)";

} // namespace

// ----------------------------------------------------------------------------
TEST ( regex, a_word_needs_a_letter_then_two_more_characters )
{
  kv::regex word( word_pattern );

  ASSERT_TRUE( word.find( "abc" ) );
  EXPECT_EQ( "abc", word.match( 0 ) );

  // Two characters is not enough: the pattern is three classes, and the
  // middle one is `+`.
  EXPECT_FALSE( word.find( "ab" ) );
  EXPECT_FALSE( word.find( "a" ) );

  EXPECT_FALSE( word.find( "_abc" ) );
  EXPECT_FALSE( word.find( "1abc" ) );
}

// ----------------------------------------------------------------------------
TEST ( regex, a_word_may_contain_a_colon_but_not_end_in_one )
{
  kv::regex word( word_pattern );

  ASSERT_TRUE( word.find( "a:b" ) );
  EXPECT_EQ( "a:b", word.match( 0 ) );

  EXPECT_FALSE( word.find( "a:" ) );
}

// ----------------------------------------------------------------------------
// The tokeniser hands it a whole line and takes the word off the front, so
// the match has to stop where the word does.
TEST ( regex, a_word_stops_at_the_first_character_it_cannot_hold )
{
  kv::regex word( word_pattern );

  ASSERT_TRUE( word.find( "process:x = 1" ) );
  EXPECT_EQ( "process:x", word.match( 0 ) );

  ASSERT_TRUE( word.find( "relativepath/to.conf" ) );
  EXPECT_EQ( "relativepath/to.conf", word.match( 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( regex, a_flag_is_bracketed_letters_at_the_start )
{
  kv::regex flag( flag_pattern );

  ASSERT_TRUE( flag.find( "[RO]" ) );
  EXPECT_EQ( "[RO]", flag.match( 0 ) );

  ASSERT_TRUE( flag.find( "[ro,local]" ) );
  EXPECT_EQ( "[ro,local]", flag.match( 0 ) );

  EXPECT_FALSE( flag.find( "[]" ) );
  EXPECT_FALSE( flag.find( "[RO" ) );

  // Anchored, so a flag has to be where a flag goes.
  EXPECT_FALSE( flag.find( "x[RO]" ) );
}

// ----------------------------------------------------------------------------
TEST ( regex, a_token_gives_its_type_and_its_name )
{
  kv::regex token( token_pattern );

  ASSERT_TRUE( token.find( "$ENV{HOME}" ) );
  EXPECT_EQ( "$ENV{HOME}", token.match( 0 ) );
  EXPECT_EQ( "ENV", token.match( 1 ) );
  EXPECT_EQ( "HOME", token.match( 2 ) );

  ASSERT_TRUE( token.find( "a $CONFIG{a.b:c} b" ) );
  EXPECT_EQ( "$CONFIG{a.b:c}", token.match( 0 ) );
  EXPECT_EQ( "CONFIG", token.match( 1 ) );
  EXPECT_EQ( "a.b:c", token.match( 2 ) );
}

// ----------------------------------------------------------------------------
// The name is optional, and an absent one is the empty string rather than a
// failure -- which is what the expander reads to mean "the whole of it".
TEST ( regex, a_token_may_have_no_name )
{
  kv::regex token( token_pattern );

  ASSERT_TRUE( token.find( "$ENV{}" ) );
  EXPECT_EQ( "ENV", token.match( 1 ) );
  EXPECT_EQ( "", token.match( 2 ) );
}

// ----------------------------------------------------------------------------
// The expander loops, so the first match has to be the leftmost one.
TEST ( regex, a_token_search_finds_the_first_of_several )
{
  kv::regex token( token_pattern );

  ASSERT_TRUE( token.find( "$ENV{HOME} and $ENV{PATH}" ) );
  EXPECT_EQ( "$ENV{HOME}", token.match( 0 ) );
  EXPECT_EQ( "HOME", token.match( 2 ) );
}

// ----------------------------------------------------------------------------
TEST ( regex, something_that_is_not_a_token_is_left_alone )
{
  kv::regex token( token_pattern );

  EXPECT_FALSE( token.find( "$9{x}" ) );
  EXPECT_FALSE( token.find( "${x}" ) );
  EXPECT_FALSE( token.find( "$ENV{not closed" ) );
}

// ----------------------------------------------------------------------------
TEST ( regex, a_colour_specification_has_five_parts )
{
  kv::regex colour( colour_pattern );

  ASSERT_TRUE( colour.find( "$person/3.5/0 0 255" ) );
  EXPECT_EQ( "person", colour.match( 1 ) );
  EXPECT_EQ( "3.5", colour.match( 2 ) );
  EXPECT_EQ( "0", colour.match( 3 ) );
  EXPECT_EQ( "0", colour.match( 4 ) );
  EXPECT_EQ( "255", colour.match( 5 ) );

  // The class name is anything but a slash, spaces included.
  ASSERT_TRUE( colour.find( "$a b/1/1 2 3" ) );
  EXPECT_EQ( "a b", colour.match( 1 ) );

  // The leading `$` is required.
  EXPECT_FALSE( colour.find( "person/3/1 2 3" ) );
}

// ----------------------------------------------------------------------------
// A group beyond what the pattern has is empty rather than an error, so a
// caller reading one too many gets nothing rather than a crash.
TEST ( regex, a_group_that_does_not_exist_is_empty )
{
  kv::regex flag( flag_pattern );

  ASSERT_TRUE( flag.find( "[RO]" ) );
  EXPECT_EQ( "", flag.match( 1 ) );
  EXPECT_EQ( "", flag.match( 99 ) );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
