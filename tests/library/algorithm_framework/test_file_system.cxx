/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief What VIAME's file system helpers do, recorded before P8-T05
/// replaces kwiversys with `std::filesystem` underneath them.
///
/// Two hundred call sites depend on these, and the expectations below were
/// not written from the documentation -- they were taken by running
/// `kwiversys::SystemTools` and writing down what came back. Four of them are
/// places `std::filesystem` would answer differently, and each is marked; a
/// call site ported directly to `std::filesystem` would have changed
/// behaviour there with nothing to catch it.

#include <viame/algorithm_framework/util/file_system.h>

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
TEST ( file_system, filename_path_drops_a_trailing_separator_first )
{
  EXPECT_EQ( "/a/b", kv::filename_path( "/a/b/c.txt" ) );
  EXPECT_EQ( "/a", kv::filename_path( "/a/b" ) );
  EXPECT_EQ( "", kv::filename_path( "c.txt" ) );
  EXPECT_EQ( "/", kv::filename_path( "/c.txt" ) );
  EXPECT_EQ( "/", kv::filename_path( "/" ) );
  EXPECT_EQ( "", kv::filename_path( "" ) );
  EXPECT_EQ( "/a/b", kv::filename_path( "/a//b///c.txt" ) );

  // **Differs from `std::filesystem`.** `path("/a/b/").parent_path()` is
  // `"/a/b"`, because the trailing separator makes an empty last component.
  // This drops the separator and then splits, so it is the *parent* of `b`.
  EXPECT_EQ( "/a", kv::filename_path( "/a/b/" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, filename_name_is_the_last_component )
{
  EXPECT_EQ( "c.txt", kv::filename_name( "/a/b/c.txt" ) );
  EXPECT_EQ( "b", kv::filename_name( "/a/b" ) );
  EXPECT_EQ( "c.txt", kv::filename_name( "c.txt" ) );
  EXPECT_EQ( "", kv::filename_name( "/a/b/" ) );
  EXPECT_EQ( "", kv::filename_name( "/" ) );
  EXPECT_EQ( "", kv::filename_name( "" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, the_extension_is_the_last_one_and_keeps_its_dot )
{
  EXPECT_EQ( ".txt", kv::filename_last_extension( "/a/b/c.txt" ) );
  EXPECT_EQ( ".gz", kv::filename_last_extension( "a.tar.gz" ) );
  EXPECT_EQ( "", kv::filename_last_extension( "a" ) );
  EXPECT_EQ( "", kv::filename_last_extension( "/a/b/" ) );

  // **Differs from `std::filesystem`.** A name that is nothing but an
  // extension is all extension here; `path(".bashrc").extension()` is empty,
  // because C++17 says a leading period with no other period is not one.
  EXPECT_EQ( ".bashrc", kv::filename_last_extension( ".bashrc" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, the_stem_is_the_name_without_that_extension )
{
  EXPECT_EQ( "c", kv::filename_without_last_extension( "/a/b/c.txt" ) );
  EXPECT_EQ( "a.tar", kv::filename_without_last_extension( "a.tar.gz" ) );
  EXPECT_EQ( "a", kv::filename_without_last_extension( "a" ) );
  EXPECT_EQ( "", kv::filename_without_last_extension( "/a/b/" ) );

  // **Differs from `std::filesystem`**, for the same reason as above:
  // `path(".bashrc").stem()` is `".bashrc"`.
  EXPECT_EQ( "", kv::filename_without_last_extension( ".bashrc" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, absolute_paths_are_recognised )
{
  EXPECT_TRUE( kv::file_is_full_path( "/a/b" ) );
  EXPECT_TRUE( kv::file_is_full_path( "/" ) );
  EXPECT_FALSE( kv::file_is_full_path( "a/b" ) );
  EXPECT_FALSE( kv::file_is_full_path( "./x" ) );
  EXPECT_FALSE( kv::file_is_full_path( "../x" ) );
  EXPECT_FALSE( kv::file_is_full_path( "" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, collapsing_resolves_dots_without_touching_the_disk )
{
  EXPECT_EQ( "/b", kv::collapse_full_path( "/a/../b" ) );
  EXPECT_EQ( "/a/b", kv::collapse_full_path( "/a/b/" ) );
  EXPECT_EQ( "/base/x", kv::collapse_full_path( "x", "/base" ) );
  EXPECT_EQ( "/base/b", kv::collapse_full_path( "a/../b", "/base" ) );

  // None of these exist, and that is not an error: this is a question about
  // the shape of a path, which is why the pipeline builder can use it on an
  // output file that has not been written yet.
  EXPECT_EQ( "/nowhere/at/all", kv::collapse_full_path( "/nowhere/at/all" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, splitting_a_path_starts_with_its_root )
{
  std::vector< std::string > parts;

  kv::split_path( "/a/b/c.txt", parts );
  ASSERT_EQ( 4u, parts.size() );
  EXPECT_EQ( "/", parts[ 0 ] );
  EXPECT_EQ( "a", parts[ 1 ] );
  EXPECT_EQ( "b", parts[ 2 ] );
  EXPECT_EQ( "c.txt", parts[ 3 ] );

  EXPECT_EQ( "/a/b/c.txt", kv::join_path( parts ) );

  // **Differs from `std::filesystem`.** A relative path still gets a first
  // element -- an empty one, where the root would be -- so that the count is
  // the same and `join_path` can tell the two apart. Iterating a
  // `std::filesystem::path` gives no such element.
  parts.clear();
  kv::split_path( "a/b", parts );
  ASSERT_EQ( 3u, parts.size() );
  EXPECT_EQ( "", parts[ 0 ] );
  EXPECT_EQ( "a", parts[ 1 ] );
  EXPECT_EQ( "b", parts[ 2 ] );
}

// ----------------------------------------------------------------------------
TEST ( file_system, unix_slashes_also_drop_a_trailing_separator )
{
  std::string path = "a\\b\\c.txt";
  kv::convert_to_unix_slashes( path );
  EXPECT_EQ( "a/b/c.txt", path );

  path = "/a/b/";
  kv::convert_to_unix_slashes( path );
  EXPECT_EQ( "/a/b", path );

  path = "/";
  kv::convert_to_unix_slashes( path );
  EXPECT_EQ( "/", path );
}

// ----------------------------------------------------------------------------
TEST ( file_system, existence_questions_answer_false_rather_than_throwing )
{
  EXPECT_FALSE( kv::file_exists( "" ) );
  EXPECT_FALSE( kv::file_exists( "/no/such/path/anywhere" ) );
  EXPECT_FALSE( kv::file_is_directory( "" ) );
  EXPECT_FALSE( kv::file_is_directory( "/no/such/path/anywhere" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, a_directory_exists_and_is_a_directory )
{
  auto const dir = kv::collapse_full_path( "fs_test_dir" );

  ASSERT_TRUE( kv::make_directory( dir ) );
  EXPECT_TRUE( kv::file_exists( dir ) );
  EXPECT_TRUE( kv::file_is_directory( dir ) );

  // A trailing separator makes no difference to either question.
  EXPECT_TRUE( kv::file_exists( dir + "/" ) );
  EXPECT_TRUE( kv::file_is_directory( dir + "/" ) );

  // Making one that is already there succeeds.
  EXPECT_TRUE( kv::make_directory( dir ) );

  EXPECT_TRUE( kv::remove_directory( dir ) );
  EXPECT_FALSE( kv::file_exists( dir ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, make_directory_makes_the_parents_too )
{
  auto const root = kv::collapse_full_path( "fs_test_tree" );
  auto const leaf = root + "/one/two/three";

  ASSERT_TRUE( kv::make_directory( leaf ) );
  EXPECT_TRUE( kv::file_is_directory( leaf ) );

  // And removing the root takes the whole tree.
  EXPECT_TRUE( kv::remove_directory( root ) );
  EXPECT_FALSE( kv::file_exists( root ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, a_file_exists_and_is_not_a_directory )
{
  auto const dir = kv::collapse_full_path( "fs_test_file_dir" );
  auto const file = dir + "/a.txt";

  ASSERT_TRUE( kv::make_directory( dir ) );
  { std::ofstream out( file ); out << "x"; }

  EXPECT_TRUE( kv::file_exists( file ) );
  EXPECT_FALSE( kv::file_is_directory( file ) );
  EXPECT_EQ( "a.txt", kv::filename_name( file ) );
  EXPECT_EQ( dir, kv::filename_path( file ) );

  EXPECT_TRUE( kv::remove_file( file ) );
  EXPECT_FALSE( kv::file_exists( file ) );

  kv::remove_directory( dir );
}

// ----------------------------------------------------------------------------
TEST ( file_system, find_file_returns_the_first_directory_that_has_it )
{
  auto const root = kv::collapse_full_path( "fs_test_find" );
  auto const first = root + "/first";
  auto const second = root + "/second";

  ASSERT_TRUE( kv::make_directory( first ) );
  ASSERT_TRUE( kv::make_directory( second ) );
  { std::ofstream out( second + "/wanted.txt" ); out << "x"; }

  auto const found = kv::find_file( "wanted.txt", { first, second } );

  EXPECT_FALSE( found.empty() );
  EXPECT_EQ( "wanted.txt", kv::filename_name( found ) );

  EXPECT_TRUE( kv::find_file( "absent.txt", { first, second } ).empty() );

  kv::remove_directory( root );
}

// ----------------------------------------------------------------------------
TEST ( file_system, listing_a_directory_includes_dot_and_dot_dot )
{
  auto const dir = kv::collapse_full_path( "fs_test_list" );

  ASSERT_TRUE( kv::make_directory( dir ) );
  { std::ofstream out( dir + "/one.txt" ); out << "x"; }
  { std::ofstream out( dir + "/two.txt" ); out << "x"; }

  auto const names = kv::directory_entries( dir );

  // Four: the two files and the two the file system always has. The plugin
  // loader and the video reader both filtered these out by suffix, which is
  // why nobody noticed they were there.
  EXPECT_EQ( 4u, names.size() );

  size_t files = 0;

  for( auto const& name : names )
  {
    if( name == "one.txt" || name == "two.txt" )
    {
      ++files;
    }
  }

  EXPECT_EQ( 2u, files );

  // Not a directory, and not an error.
  EXPECT_TRUE( kv::directory_entries( dir + "/one.txt" ).empty() );
  EXPECT_TRUE( kv::directory_entries( "/no/such/path" ).empty() );

  kv::remove_directory( dir );
}

// ----------------------------------------------------------------------------
TEST ( file_system, the_environment_says_whether_it_had_the_name )
{
  setenv( "VIAME_FS_TEST_VAR", "a value", 1 );

  std::string value;
  EXPECT_TRUE( kv::get_env( "VIAME_FS_TEST_VAR", value ) );
  EXPECT_EQ( "a value", value );

  unsetenv( "VIAME_FS_TEST_VAR" );

  // The out parameter is left alone when the name is not set, so a caller
  // that seeded it with a default keeps the default.
  value = "untouched";
  EXPECT_FALSE( kv::get_env( "VIAME_FS_TEST_VAR", value ) );
  EXPECT_EQ( "untouched", value );

  EXPECT_EQ( nullptr, kv::get_env( "VIAME_FS_TEST_VAR" ) );
}

// ----------------------------------------------------------------------------
TEST ( file_system, the_working_directory_is_absolute )
{
  auto const cwd = kv::current_working_directory();

  EXPECT_FALSE( cwd.empty() );
  EXPECT_TRUE( kv::file_is_full_path( cwd ) );
  EXPECT_TRUE( kv::file_is_directory( cwd ) );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );

  // Somewhere of its own to make and remove directories in, so that a failed
  // run leaves its mess in the build tree rather than wherever ctest started.
  kwiver::vital::make_directory( VIAME_FS_TEST_DIR );

  if( chdir( VIAME_FS_TEST_DIR ) != 0 )
  {
    std::cerr << "could not enter " << VIAME_FS_TEST_DIR << "\n";
    return 1;
  }

  return RUN_ALL_TESTS();
}
