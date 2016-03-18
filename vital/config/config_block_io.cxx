/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief config_block IO operations implementation
 */

#include "config_block_io.h"
#include "config_block_exception.h"
#include "config_parser.h"

#include <vital/logger/logger.h>

#include <vital/util/tokenize.h>

#include <vital/vital_foreach.h>

#include <kwiversys/SystemTools.hxx>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <list>

#if defined(_WIN32)
#include <shlobj.h>
#endif

namespace kwiver {
namespace vital {

namespace {

#if defined(_WIN32)
// ------------------------------------------------------------------
// Helper method to add known special paths to a path list
void add_windows_path( config_path_list_t & paths, int which )
{
  char buffer[MAX_PATH];
  if ( SHGetFolderPath ( 0, which, 0, 0, buffer ) )
  {
    auto path = config_path_t{ buffer };
    kwiversys::SystemTools::ConvertToUnixSlashes( path );
    paths.push_back( path );
  }
}
#endif


// ------------------------------------------------------------------
// Helper method to get application specific paths from generic paths
config_path_list_t
application_paths( config_path_list_t const& paths,
                   std::string const& application_name,
                   std::string const& application_version )
{
  auto result = config_path_list_t{};
  VITAL_FOREACH ( auto const& path, paths )
  {
    auto const& app_path = path + "/" + application_name;

    if ( ! application_version.empty() )
    {
      result.push_back( app_path + "/" + application_version );
    }
    result.push_back( app_path );
  }

  return result;
}


// ------------------------------------------------------------------
// Helper method to write out a comment to a configuration file ostream
/**
 * Makes sure there is no trailing white-space printed to file.
 */
void
write_cb_comment( std::ostream& ofile, config_block_description_t const& comment )
{
  typedef config_block_description_t cbd_t;
  size_t line_width = 80;
  cbd_t comment_token = cbd_t( "#" );

  // Add a leading new-line to separate comment block from previous config
  // entry.
  ofile << "\n";

  // preserve manually specified new-lines in the comment string, adding a
  // trailing new-line
  std::list< cbd_t > blocks;
  tokenize( comment, blocks, "\n" );
  while ( blocks.size() > 0 )
  {
    cbd_t cur_block = blocks.front();
    blocks.pop_front();

    // Comment lines always start with the comment token
    cbd_t line_buffer = comment_token;

    // Counter of additional spaces to place in front of the next non-empty
    // word added to the line buffer. There is always at least one space
    // between words.
    size_t spaces = 1;

    std::list< cbd_t > words;
    // Not using token-compress in case there is purposeful use of multiple
    // adjacent spaces, like in bullited lists. This, however, leaves open
    // the appearance of empty-string words in the loop, which are handled.
    tokenize( cur_block, words );
    while ( words.size() > 0 )
    {
      cbd_t cur_word = words.front();
      words.pop_front();

      // word is an empty string, meaning an intentional space was encountered.
      if ( cur_word.size() == 0 )
      {
        ++spaces;
      }
      else
      {
        if ( ( line_buffer.size() + spaces + cur_word.size() ) > line_width )
        {
          ofile << line_buffer << "\n";
          line_buffer = comment_token;
          // On a line split, it makes sense to me that leading spaces are
          // treated as trailing white-space, which should not be output.
          spaces = 1;
        }
        line_buffer += std::string( spaces, ' ' ) + cur_word;
        spaces = 1;
      }
    }

    // flush remaining contents of line buffer if there is anything
    if ( line_buffer.size() > 0 )
    {
      ofile << line_buffer << "\n";
    }
  }
} // write_cb_comment

} //end anonymous namespace


// ------------------------------------------------------------------
// Helper method to get all possible locations of application config files
config_path_list_t
config_file_paths( std::string const& application_name,
                   std::string const& application_version,
                   config_path_t const& install_prefix )
{
  // First, add any paths specified by our local environment variable
  auto paths = config_path_list_t{};
  kwiversys::SystemTools::GetPath( paths, "KWIVER_CONFIG_PATH" );

  // Now add platform specific directories
  auto data_paths = config_path_list_t{};

#if defined(_WIN32)

  // Add the application data directories
  add_windows_path( data_paths, CSIDL_LOCAL_APPDATA );
  add_windows_path( data_paths, CSIDL_APPDATA );
  add_windows_path( data_paths, CSIDL_COMMON_APPDATA );

#else

  auto const home = kwiversys::SystemTools::GetEnv( "HOME" );

# if defined(__APPLE__)
  if ( home && *home )
  {
    data_paths.push_back(
      config_path_t( home ) + "/Library/Application Support" );
  }
  data_paths.push_back( "/Library/Application Support" );
# endif

  // Get the list of configuration data paths
  auto config_paths = config_path_list_t{};
  kwiversys::SystemTools::GetPath( config_paths, "XDG_CONFIG_HOME" );
  if ( home && *home )
  {
    config_paths.push_back( config_path_t( home ) + "/.config" );
  }
  config_paths.push_back( "/etc/xdg" );
  config_paths.push_back( "/etc" );

  // Add application information to config paths and append to paths
  config_paths = application_paths(
                   config_paths, application_name, application_version );
  paths.insert( paths.end(), config_paths.begin(), config_paths.end() );

  // Get the list of application data paths
  data_paths.push_back( "/usr/local/share" );
  data_paths.push_back( "/usr/share" );

#endif

  // Add install-local data path if install prefix is not a standard prefix
  auto const nonstandard_prefix =
    !install_prefix.empty() &&
    install_prefix != "/usr" &&
    install_prefix != "/usr/local";
  if ( nonstandard_prefix )
  {
    data_paths.push_back( install_prefix + "/share" );
  }

  // Turn the generic FHS data paths into application data paths...
  data_paths = application_paths(
                 data_paths, application_name, application_version );

  // ...then into config paths and add to final list
  VITAL_FOREACH ( auto const& path, data_paths )
  {
    paths.push_back( path + "/config" );
  }

  // Add install-local config paths if install prefix is not a standard prefix
  if ( nonstandard_prefix )
  {
    paths.push_back( install_prefix + "/share/config" );
    paths.push_back( install_prefix + "/config" );
#if defined(__APPLE__)
    paths.push_back( install_prefix + "/Resources/config" );
#endif
  }

  return paths;
}


// ------------------------------------------------------------------
config_block_sptr
read_config_file( config_path_t const&     file_path,
                  config_path_list_t const& search_path )
{
  // The file specified really must be a file.
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    throw config_file_not_found_exception( file_path,
          "File does not exist." );
  }

  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw config_file_not_found_exception( file_path,
          "Path given doesn't point to a regular file." );
  }

  kwiver::vital::config_parser the_parser;
  the_parser.add_search_path( search_path );
  the_parser.parse_config( file_path );

  return the_parser.get_config();
}


// ------------------------------------------------------------------
config_block_sptr
read_config_file( std::string const& file_name,
                  std::string const& application_name,
                  std::string const& application_version,
                  config_path_t const& install_prefix,
                  bool merge )
{
  auto logger = logger_handle_t{ get_logger( "read_config_file" ) };

  auto result = config_block_sptr{};

  auto const& search_paths =
    config_file_paths( application_name, application_version, install_prefix );

  // See if file name is an absolute path. If so, then just process the file.
  if ( kwiversys::SystemTools::FileIsFullPath( file_name ) )
  {
    // The file is on a absolute path.
    auto const& config = read_config_file( file_name, search_paths );
    return config;
  }

  // use current directory when searching for this file.
  config_path_list_t local_search_paths( search_paths );
  local_search_paths.push_back( "." );

  // File name is relative, so go through the search process.
  VITAL_FOREACH( auto const& search_path, local_search_paths )
  {
    auto const& config_path = search_path + "/" + file_name;

    // Check that file exists. We need this check here because when
    // the parser does not find a file it throws and we want to try
    // the next directory if file not found.
    //
    // Cant use the parsers exception as an indication of a bad file
    // because the parser will throw the same exception if an include
    // file is not found.
    if ( ! kwiversys::SystemTools::FileExists( config_path ) ||
         kwiversys::SystemTools::FileIsDirectory( config_path ) )
    {
      continue;
    }

    auto const& config = read_config_file( config_path, search_paths );

    LOG_DEBUG( logger, "Read config file \"" << config_path << "\"" );

    if ( ! merge )
    {
      return config;
    }
    else if ( result )
    {
      // Merge under current configuration
      config->merge_config( result );
    }

    // Continue with new config
    result = config;
  } // end foreach

  // Throw file-not-found if we ran out of paths without finding anything
  if ( ! result )
  {
    throw config_file_not_found_exception(
      file_name, "No matching file found in the search paths." );
  }

  return result;
}


// ------------------------------------------------------------------
// Output to file the given \c config_block object to the specified file path
void
write_config_file( config_block_sptr const& config,
                   config_path_t const&     file_path )
{
  using std::cerr;
  using std::endl;

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw config_file_write_exception( file_path,
          "Path given is a directory, to which we clearly can't write." );
  }

  // Check that the directory of the given filepath exists, creating necessary
  // directories where needed.
  config_path_t parent_dir = kwiversys::SystemTools::GetFilenamePath(
    kwiversys::SystemTools::CollapseFullPath( file_path ) );
  if ( ! kwiversys::SystemTools::FileIsDirectory( parent_dir ) )
  {
    //std::cerr << "at least one containing directory not found, creating them..." << std::endl;
    if ( ! kwiversys::SystemTools::MakeDirectory( parent_dir ) )
    {
      throw config_file_write_exception( parent_dir,
            "Attempted directory creation, but no directory created! No idea what happened here..." );
    }
  }

  // open output file and write each key/value to a line.
  std::ofstream ofile( file_path.c_str() );

  write_config( config, ofile );
  ofile.close();
}


// ------------------------------------------------------------------
void write_config( config_block_sptr const& config,
                   std::ostream&            ofile )
{
  // If there are no config parameters in the given config_block, throw
  if ( ! config->available_values().size() )
  {
    throw config_file_write_exception( "<stream>",
          "No parameters in the given config_block!" );
  }

  // Gather available keys and sort them alphanumerically for a sensibly layout
  // file.
  config_block_keys_t avail_keys = config->available_values();
  std::sort( avail_keys.begin(), avail_keys.end() );


  bool prev_had_descr = false;  // for additional spacing
  VITAL_FOREACH( config_block_key_t key, avail_keys )
  {
    // Each key may or may not have an associated description string. If there
    // is one, write that out as a comment.
    // - comments will be limited to 80 character width lines, including "# "
    //   prefix.
    // - value output format: "key_path = value\n"

    config_block_description_t descr = config->get_description( key );

    if ( descr != config_block_description_t() )
    {
      //std::cerr << "[write_config_file] Writing comment for '" << key << "'." << std::endl;
      write_cb_comment( ofile, descr );
      prev_had_descr = true;
    }
    else if ( prev_had_descr )
    {
      // Add a spacer line after a k/v with a description
      ofile << "\n";
      prev_had_descr = false;
    }

    ofile << key << " = " << config->get_value< config_block_value_t > ( key ) << "\n";

    std::string file;
    int line;
    if ( config->get_location( key, file, line ) )
    {
      ofile << "# defined - " << file << ":" << line << "\n";
    }
  }
  ofile.flush();
} // write_config_file

} }   // end namespace
