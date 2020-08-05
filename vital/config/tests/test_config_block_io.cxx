/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief core config_block_io tests
 */

#include <test_gtest.h>
#include <test_tmpfn.h>

#include <vital/vital_types.h>
#include <vital/config/config_block_io.h>

#include <kwiversys/SystemTools.hxx>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

kwiver::vital::config_path_t g_data_dir;

using namespace kwiver::vital;
typedef kwiversys::SystemTools ST;

// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class config_block_io : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F( config_block_io, config_path_not_exist )
{
  path_t fp( "/this/shouldnt/exist/anywhere" );

  EXPECT_THROW(
    kwiver::vital::read_config_file( fp ),
    kwiver::vital::config_file_not_found_exception )
    << "Calling read_config_file with non-existent file";
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, config_path_not_file )
{
  path_t fp = ST::GetCurrentWorkingDirectory();

  EXPECT_THROW(
    kwiver::vital::read_config_file( fp ),
    kwiver::vital::config_file_not_found_exception )
    << "calling read_config_file with directory path as argument";
}

// ----------------------------------------------------------------------------
static
void
print_config( config_block_sptr const& config, bool include_location = false,
              char const* message = "Available keys in the config_block" )
{
  using std::cerr;
  using std::endl;

  cerr << message << ":" << endl;
  for ( auto const& key : config->available_values() )
  {
    cerr << "\t\"" << key << "\""
         << ( config->is_read_only( key ) ? " [RO]" : "" )
         << " := \"" << config->get_value< std::string > ( key ) << "\""
         << endl;
    if (include_location)
    {
      std::string file( "undefined" );
      int line(0);
      config->get_location( key, file, line );

      cerr << "\t   Defined at: " << file << ":" << line
           << endl;
    }
  }
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, successful_config_read )
{
  config_block_sptr config = kwiver::vital::read_config_file( data_dir + "/test_config-valid_file.txt" );

  print_config( config, true );

  EXPECT_EQ( 25,
             config->available_values().size() );
  EXPECT_EQ( "baz",
             config->get_value< std::string > ( "foo:bar" ) );
  EXPECT_EQ( "stuff",
             config->get_value< std::string > ( "foo:things" ) );
  EXPECT_EQ( "cool things and stuff",
             config->get_value< std::string > ( "foo:sublevel:value" ) );
  EXPECT_EQ( "a value    with  spaces",
             config->get_value< std::string > ( "second_block:has" ) );
  EXPECT_EQ( "has a trailing comment",
             config->get_value< std::string > ( "second_block:more" ) );
  EXPECT_NEAR( 3.14159,
               config->get_value< float > ( "global_var" ),
               0.000001 );

  EXPECT_TRUE( config->is_read_only( "global_var" ) );

  EXPECT_NEAR( 1.12,
               config->get_value< double > ( "global_var2" ),
               0.000001 );
  EXPECT_EQ( "should be valid",
             config->get_value< std::string > ( "tabbed:value" ) );

  // extract sub-block, see that value access maintained
  config_block_sptr foo_subblock = config->subblock_view( "foo" );
  EXPECT_EQ( "baz",
             foo_subblock->get_value< std::string > ( "bar" ) );
  EXPECT_EQ( "cool things and stuff",
             foo_subblock->get_value< std::string > ( "sublevel:value" ) );
  EXPECT_EQ( "cool things and stuff",
             config->subblock_view( "foo" )->subblock_view( "sublevel" )
                ->get_value< std::string > ( "value" ) );

  EXPECT_EQ( "new value",
             config->get_value< std::string > ( "local" ) )
    << "Value from macro";

  EXPECT_EQ( "a value    with  spaces",
             config->get_value< std::string > ( "new_block:next_level:has" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, successful_config_read_named_block )
{
  config_block_sptr config = kwiver::vital::read_config_file( data_dir + "/test_config-valid_file.txt" );

  print_config( config );

  EXPECT_EQ( 25,
             config->available_values().size() );
  EXPECT_EQ( "baz",
             config->get_value< std::string > ( "foo:bar" ) );
  EXPECT_EQ( "stuff",
             config->get_value< std::string > ( "foo:things" ) );
  EXPECT_EQ( "cool things and stuff",
             config->get_value< std::string > ( "foo:sublevel:value" ) );
  EXPECT_EQ( "a value    with  spaces",
             config->get_value< std::string > ( "second_block:has" ) );
  EXPECT_EQ( "has a trailing comment",
             config->get_value< std::string > ( "second_block:more" ) );
  EXPECT_NEAR( 3.14159,
               config->get_value< float > ( "global_var" ),
               0.000001 );
  EXPECT_NEAR( 1.12,
               config->get_value< double > ( "global_var2" ),
               0.000001 );
  EXPECT_EQ( "should be valid",
             config->get_value< std::string > ( "tabbed:value" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, include_files )
{
  config_block_sptr config = kwiver::vital::read_config_file( data_dir + "/test_config-include-a.txt" );

  print_config( config );

  EXPECT_EQ( 6,
             config->available_values().size() );
  EXPECT_EQ( "outer",
             config->get_value< std::string > ( "a:var" ) );
  EXPECT_EQ( "val",
             config->get_value< std::string > ( "outer_block:b:key" ) );
  EXPECT_EQ( "on",
             config->get_value< std::string > ( "outer_block:general:logging" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, include_files_in_path )
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  char const *pathSep = ";";
#else
  char const *pathSep = ":";
#endif

  // Should pick up file in the first directory in the path list.
  kwiversys::SystemTools::PutEnv(
    "KWIVER_CONFIG_PATH=" + data_dir + "/test_config-standard-dir-first"
                + pathSep + data_dir + "/test_config-standard-dir-second" );
  auto config_a =
    kwiver::vital::read_config_file( data_dir + "/test_config-include_files_in_path.txt" );
  EXPECT_EQ( 1,
             config_a->available_values().size() );
  EXPECT_EQ( "a",
             config_a->get_value< std::string >( "included:a" ) );

  // If we set the KCP in reverse order, we should see
  kwiversys::SystemTools::PutEnv(
    "KWIVER_CONFIG_PATH=" + data_dir + "/test_config-standard-dir-second"
                + pathSep + data_dir + "/test_config-standard-dir-first");
  auto config_b =
    kwiver::vital::read_config_file( data_dir + "/test_config-include_files_in_path.txt" );
  EXPECT_EQ( 2,
             config_b->available_values().size() );
  EXPECT_EQ( "b",
             config_b->get_value< std::string >( "included:a" ) );
  EXPECT_EQ( "c",
             config_b->get_value< std::string >( "included:b" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, include_files_failure )
{
  // NOT setting KWIVER_CONFIG_PATH

  // Should pick up file with the same name in the first directory in the
  // path list: test_config-standard-dir-first/test_config-standard.txt
  EXPECT_THROW(
    kwiver::vital::read_config_file( data_dir + "/test_config-include_files_in_path.txt" ),
    config_file_not_found_exception )
    << "Included file not in search path dirs";
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, invalid_config_file )
{
  EXPECT_THROW(
    kwiver::vital::read_config_file( data_dir + "/test_config-invalid_file.txt" ),
    kwiver::vital::config_file_not_parsed_exception )
    << "Calling config_block read on badly formatted file";
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, invalid_keypath )
{
  EXPECT_THROW(
    kwiver::vital::read_config_file( data_dir + "/test_config-invalid_keypath.txt" ),
    kwiver::vital::config_file_not_parsed_exception )
    << "Read attempt on file with invalid key path";
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, config_with_comments )
{
  config_block_sptr config = kwiver::vital::read_config_file( data_dir + "/test_config-comments.txt" );

  using std::string;

  EXPECT_EQ( 4,
             config->available_values().size() );

  EXPECT_EQ( "on",
             config->get_value< string > ( "general:logging" ) );
  EXPECT_EQ( "foo",
             config->get_value< string > ( "general:another_var" ) );
  EXPECT_EQ( "bar",
             config->get_value< string > ( "general:yet_more" ) );
  EXPECT_EQ( "things and stuff",
             config->get_value< string > ( "final:value" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, write_config_simple_success )
{
  using namespace kwiver;
  using namespace std;

  config_block_sptr orig_config = config_block::empty_config( "simple_test" );

  config_block_key_t keyA = config_block_key_t( "test_key_1" );
  config_block_key_t keyB = config_block_key_t( "test_key_2" );
  config_block_key_t keyC = config_block_key_t( "test_key_3" );
  config_block_key_t keyD = config_block_key_t( "test_key_4" );
  config_block_key_t keyE = config_block_key_t( "test_key_5" );
  config_block_key_t keyF = config_block_key_t( "test_key_6" );
  config_block_key_t keyG = config_block_key_t( "test_key_7" );

  config_block_value_t valueA = config_block_value_t( "test_value_a" );
  config_block_value_t valueB = config_block_value_t( "test_value_b" );
  config_block_value_t valueC = config_block_value_t( "test_value_c" );
  config_block_value_t valueD = config_block_value_t( "test_value_d" );
  config_block_value_t valueE = config_block_value_t( "test_value_e" );
  config_block_value_t valueF = config_block_value_t( "test_value_f" );
  config_block_value_t valueG = config_block_value_t( "test_value_g" );

  config_block_description_t descrD = config_block_description_t( "Test descr 1" );
  config_block_description_t descrE = config_block_description_t(
    "This is a really long description that should probably span multiple "
    "lines because it exceeds the defined character width we would like "
    "in order to make output files more readable."
                                                                );
  config_block_description_t descrF = config_block_description_t(
    "this is a comment\n"
    "that has manual new-line splits\n"
    "that should be preserved\n"
    "\n"
    "Pretend list:\n"
    "  - foo\n"
    "    - bar\n"
                                                                );
  config_block_description_t descrG = config_block_description_t(
    "This has a # in it"
                                                                );

  config_block_key_t subblock_name = config_block_key_t( "subblock" );
  config_block_sptr subblock = orig_config->subblock_view( subblock_name );

  orig_config->set_value( keyA, valueA );
  orig_config->set_value( keyB, valueB );
  subblock->set_value( keyC, valueC );
  orig_config->set_value( keyD, valueD, descrD );
  orig_config->set_value( keyE, valueE, descrE );
  orig_config->set_value( keyF, valueF, descrF );
  orig_config->set_value( keyG, valueG, descrG );

  print_config( orig_config, false, "ConfigBlock for writing" );

  auto const& output_path_1 =
    kwiver::testing::temp_file_name("test_config_output_1-", ".conf");

  cerr << "Writing config_block to: " << output_path_1 << endl;
  write_config_file( orig_config, std::string( output_path_1 ) );

  auto const& output_path_2 =
    kwiver::testing::temp_file_name("test_config_output_2-", ".conf");

  cerr << "Writing config_block to: " << output_path_2 << endl;
  write_config_file( orig_config, std::string( output_path_2 ) );

  // Read files back in, confirning output is readable and the same as
  // what we should have output.
  std::vector< config_block_sptr > configs;
  configs.push_back( read_config_file( output_path_1 ) );
  configs.push_back( read_config_file( output_path_2 ) );

  for ( config_block_sptr config : configs )
  {
    EXPECT_EQ( 7, config->available_values().size() );
    EXPECT_EQ( valueA,
             config->get_value< config_block_value_t > ( keyA ) );
    EXPECT_EQ( valueB,
             config->get_value< config_block_value_t > ( keyB ) );
    EXPECT_EQ( valueC,
             config->get_value< config_block_value_t > ( subblock_name +
               config_block::block_sep() + keyC ) );
    EXPECT_EQ( valueD,
             config->get_value< config_block_value_t > ( keyD ) );
    EXPECT_EQ( valueE,
             config->get_value< config_block_value_t > ( keyE ) );
    EXPECT_EQ( valueF,
             config->get_value< config_block_value_t > ( keyF ) );
    EXPECT_EQ( valueG,
             config->get_value< config_block_value_t > ( keyG ) );
  }

  // Clean up generated configuration files
  if( ! ST::RemoveFile( output_path_1 ) )
  {
    cerr << "Failed to remove output path 1" << endl;
}
  if( ! ST::RemoveFile( output_path_2 ) )
  {
    cerr << "Failed to remove output path 2" << endl;
  }
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, invalid_directory_write )
{
  using namespace kwiver;
  config_block_sptr config = config_block::empty_config( "empty" );
  config->set_value( "foo", "bar" );
  EXPECT_THROW(
    write_config_file( config, data_dir ),
    config_file_write_exception )
    << "Attempting write on a directory path";
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, empty_config_write_failure )
{
  using namespace kwiver;
  using namespace std;

  config_block_sptr config = config_block::empty_config( "empty" );
  auto const& output_path =
    kwiver::testing::temp_file_name("test_config_output_empty-", ".conf");

  EXPECT_THROW(
    write_config_file( config, output_path ),
    config_file_write_exception )
    << "Attempting write of a config with nothing in it";

  // If the test failed, clean-up the file created.
  if ( 0 == ST::RemoveFile( output_path ) )
  {
    cerr << "Test failed and output file created. Removing." << endl;
  }
}

// ----------------------------------------------------------------------------
/// Return KWIVER_CONFIG_PATH string referring to two sub-directories in dir
static
std::string
test_standard_paths( kwiver::vital::config_path_t const& data_dir )
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  char const *pathSep = ";";
#else
  char const *pathSep = ":";
#endif

  return data_dir + "/test_config-standard-dir-first" + pathSep +
         data_dir + "/test_config-standard-dir-second";
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, standard_config_read_without_merge )
{
  kwiversys::SystemTools::PutEnv(
    "KWIVER_CONFIG_PATH=" + test_standard_paths( data_dir ) );

  std::cerr << "Current working directory: "
            << kwiversys::SystemTools::GetCurrentWorkingDirectory()
            << std::endl;
  auto const config =
    kwiver::vital::read_config_file( "test_config-standard.txt",
                                     "vital", {}, {}, false );

  EXPECT_EQ( 2,
             config->available_values().size() );

  EXPECT_EQ( 1,
             config->get_value< int >( "general:zero" ) );

  EXPECT_EQ( "foo",
             config->get_value< std::string > ( "general:first" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, standard_config_read_without_merge_with_cwd )
{
  kwiversys::SystemTools::ChangeDirectory( data_dir );
  kwiversys::SystemTools::PutEnv(
    "KWIVER_CONFIG_PATH=" + test_standard_paths( data_dir ) );

  std::cerr << "Current working directory: "
            << kwiversys::SystemTools::GetCurrentWorkingDirectory()
            << std::endl;

  auto const config =
    kwiver::vital::read_config_file( "test_config-standard.txt",
                                     "vital", {}, {}, false );

  EXPECT_EQ( 1,
             config->available_values().size() );

  EXPECT_EQ( 0,
             config->get_value< int > ( "general:zero" ) );
}


// ----------------------------------------------------------------------------
TEST_F( config_block_io, standard_config_read_with_merge )
{
  kwiversys::SystemTools::PutEnv(
    "KWIVER_CONFIG_PATH=" + test_standard_paths( data_dir ) );

  std::cerr << "Current working directory: "
            << kwiversys::SystemTools::GetCurrentWorkingDirectory()
            << std::endl;
  auto const config =
    kwiver::vital::read_config_file( "test_config-standard.txt",
                                     "vital", {}, {}, true );
  EXPECT_EQ( 3,
             config->available_values().size() );

  EXPECT_EQ( "foo",
             config->get_value< std::string > ( "general:first" ) );

  EXPECT_EQ( "bar",
             config->get_value< std::string > ( "general:second" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, standard_config_read_with_merge_with_cwd )
{
  kwiversys::SystemTools::ChangeDirectory( data_dir );
  kwiversys::SystemTools::PutEnv(
    "KWIVER_CONFIG_PATH=" + test_standard_paths( data_dir ) );

  std::cerr << "Current working directory: "
            << kwiversys::SystemTools::GetCurrentWorkingDirectory()
            << std::endl;

  auto const config =
    kwiver::vital::read_config_file( "test_config-standard.txt",
                                     "vital", {}, {}, true );

  EXPECT_EQ( 3,
             config->available_values().size() );

  EXPECT_EQ( 0,
             config->get_value< int > ( "general:zero" ) );

  EXPECT_EQ( "foo",
             config->get_value< std::string > ( "general:first" ) );

  EXPECT_EQ( "bar",
             config->get_value< std::string > ( "general:second" ) );
}

// ----------------------------------------------------------------------------
TEST_F( config_block_io, standard_config_read_from_prefix )
{
  auto const config =
    kwiver::vital::read_config_file( "test_config-standard.txt",
                                     "vital", "test", data_dir );
  std::cerr << "Current working directory: "
            << kwiversys::SystemTools::GetCurrentWorkingDirectory()
            << std::endl;

  EXPECT_EQ( 3,
             config->available_values().size() );

  EXPECT_EQ( "woof",
             config->get_value< std::string > ( "animal:dog" ) );

  EXPECT_EQ( "meow",
             config->get_value< std::string > ( "animal:cat" ) );

  EXPECT_EQ( "oink",
             config->get_value< std::string > ( "animal:pig" ) );
}
