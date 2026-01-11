/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "utilities_file.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>

#if WIN32 || ( __cplusplus >= 201703L && __has_include(<filesystem>) )
  #include <filesystem>
  namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace filesystem = std::experimental::filesystem;
#endif

namespace viame {

// =============================================================================
// Filesystem utilities
// =============================================================================

bool does_file_exist( const std::string& location )
{
  return filesystem::exists( location ) &&
         !filesystem::is_directory( location );
}

bool does_folder_exist( const std::string& location )
{
  return filesystem::exists( location ) &&
         filesystem::is_directory( location );
}

bool list_all_subfolders( const std::string& location,
                          std::vector< std::string >& subfolders )
{
  subfolders.clear();

  if( !does_folder_exist( location ) )
  {
    return false;
  }

  filesystem::path dir( location );

  for( filesystem::directory_iterator dir_iter( dir );
       dir_iter != filesystem::directory_iterator();
       ++dir_iter )
  {
    if( filesystem::is_directory( *dir_iter ) )
    {
      subfolders.push_back( dir_iter->path().string() );
    }
  }

  return true;
}

bool list_files_in_folder( std::string location,
                           std::vector< std::string >& filepaths,
                           bool search_subfolders,
                           std::vector< std::string > extensions )
{
  filepaths.clear();

  if( !does_folder_exist( location ) )
  {
    return false;
  }

#ifdef WIN32
  if( location.back() != '\\' )
  {
    location = location + "\\";
  }
#else
  if( location.back() != '/' )
  {
    location = location + "/";
  }
#endif

  filesystem::path dir( location );

  for( filesystem::directory_iterator file_iter( dir );
       file_iter != filesystem::directory_iterator();
       ++file_iter )
  {
    if( filesystem::is_regular_file( *file_iter ) )
    {
      if( extensions.empty() )
      {
        filepaths.push_back( file_iter->path().string() );
      }
      else
      {
        for( unsigned i = 0; i < extensions.size(); i++ )
        {
          if( file_iter->path().extension() == extensions[i] )
          {
            filepaths.push_back( file_iter->path().string() );
            break;
          }
        }
      }
    }
    else if( filesystem::is_directory( *file_iter ) && search_subfolders )
    {
      std::vector< std::string > subfiles;
      list_files_in_folder( file_iter->path().string(),
        subfiles, search_subfolders, extensions );

      filepaths.insert( filepaths.end(), subfiles.begin(), subfiles.end() );
    }
  }

  return true;
}

bool create_folder( const std::string& location )
{
  filesystem::path dir( location );

  if( !filesystem::exists( dir ) )
  {
    return filesystem::create_directories( dir );
  }

  return false;
}

bool folder_contains_less_than_n_files( const std::string& folder, unsigned n )
{
  auto dir = filesystem::directory_iterator( folder );
  unsigned count = 0;

  for( auto i : dir )
  {
    (void)i; // Suppress unused variable warning
    count++;

    if( count >= n )
    {
      return false;
    }
  }

  return true;
}

// =============================================================================
// Path manipulation utilities
// =============================================================================

std::string append_path( const std::string& p1, const std::string& p2 )
{
  return p1 + "/" + p2;
}

std::string get_filename_no_path( const std::string& path )
{
  return filesystem::path( path ).filename().string();
}

std::string get_filename_with_last_path( const std::string& path )
{
  return append_path( filesystem::path( path ).parent_path().filename().string(),
                      filesystem::path( path ).filename().string() );
}

std::string replace_ext_with( const std::string& file_name, const std::string& ext )
{
  return file_name.substr( 0, file_name.find_last_of( '.' ) ) + ext;
}

std::string add_ext_unto( const std::string& path, const std::string& ext )
{
  if( !path.empty() && ( path.back() == '/' || path.back() == '\\' ) )
  {
    return path.substr( 0, path.size() - 1 ) + ext;
  }

  return path + ext;
}

std::string add_aux_ext( const std::string& file_name, unsigned id )
{
  std::size_t last_index = file_name.find_last_of( "." );
  std::string file_name_no_ext = file_name.substr( 0, last_index );
  std::string aux_addition = "_aux";

  if( id > 1 )
  {
    aux_addition += std::to_string( id );
  }

  return file_name_no_ext + aux_addition + file_name.substr( last_index );
}

bool ends_with_extension( const std::string& str, const std::string& ext )
{
  if( str.length() >= ext.length() )
  {
    return( 0 == str.compare( str.length() - ext.length(),
                              ext.length(), ext ) );
  }
  else
  {
    return false;
  }
}

bool ends_with_extension( const std::string& str,
                          const std::vector< std::string >& exts )
{
  for( const auto& ext : exts )
  {
    if( ends_with_extension( str, ext ) )
    {
      return true;
    }
  }
  return false;
}

std::string add_quotes( const std::string& str )
{
  return "\"" + str + "\"";
}

// =============================================================================
// String parsing utilities
// =============================================================================

void string_to_vector( const std::string& str,
                       std::vector< std::string >& out,
                       const std::string delims )
{
  out.clear();

  std::stringstream ss( str );
  std::string line;

  while( std::getline( ss, line ) )
  {
    std::size_t prev = 0, pos;
    while( ( pos = line.find_first_of( delims, prev ) ) != std::string::npos )
    {
      if( pos > prev )
      {
        std::string word = line.substr( prev, pos - prev );
        if( !word.empty() )
        {
          out.push_back( word );
        }
      }
      prev = pos + 1;
    }
    if( prev < line.length() )
    {
      std::string word = line.substr( prev, std::string::npos );
      if( !word.empty() )
      {
        out.push_back( word );
      }
    }
  }
}

void string_to_set( const std::string& str,
                    std::unordered_set< std::string >& out,
                    const std::string delims )
{
  out.clear();
  std::vector< std::string > tmp;
  string_to_vector( str, tmp, delims );
  out.insert( tmp.begin(), tmp.end() );
}

// =============================================================================
// File reading utilities
// =============================================================================

bool file_to_vector( const std::string& fn,
                     std::vector< std::string >& out,
                     bool reset )
{
  std::ifstream in( fn.c_str() );

  if( reset )
  {
    out.clear();
  }

  if( !in )
  {
    std::cerr << "Unable to open " << fn << std::endl;
    return false;
  }

  std::string line;
  while( std::getline( in, line ) )
  {
    if( !line.empty() )
    {
      out.push_back( line );
    }
  }
  return true;
}

bool load_file_list( const std::string& file,
                     std::vector< std::string >& output )
{
  std::ifstream fin( file );
  output.clear();

  if( !fin )
  {
    return false;
  }

  while( !fin.eof() )
  {
    std::string line;
    std::getline( fin, line );
    output.push_back( line );
  }

  fin.close();
  return true;
}

bool file_contains_string( const std::string& file, const std::string& key )
{
  std::ifstream fin( file );
  while( !fin.eof() )
  {
    std::string line;
    std::getline( fin, line );

    if( line.find( key ) != std::string::npos )
    {
      fin.close();
      return true;
    }
  }
  fin.close();
  return false;
}

double get_file_frame_rate( const std::string& file )
{
  std::ifstream fin( file );

  if( !fin )
  {
    return -1.0;
  }

  std::string number;

  for( unsigned i = 0; i < 4 && !fin.eof(); i++ )
  {
    std::string line;
    std::getline( fin, line );

    if( line.size() > 5 && line[0] == '#' )
    {
      for( unsigned p = 0; p < line.size() - 4; p++ )
      {
        if( line.substr( p, 4 ) == "fps:" || line.substr( p, 4 ) == "fps=" )
        {
          for( unsigned l = p + 4; l < line.size(); l++ )
          {
            if( line[l] == ' ' )
            {
              continue;
            }
            else if( std::isdigit( line[l] ) || line[l] == '.' )
            {
              number = number + line[l];
            }
            else
            {
              break;
            }
          }
        }
      }
    }
  }

  fin.close();

  if( number.empty() )
  {
    return -1.0;
  }

  return std::stof( number );
}

} // end namespace viame
