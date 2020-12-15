/*ckwg +29
 * Copyright 2011-2018, 2020 by Kitware, Inc.
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

#include "pipeline_builder.h"

#if defined(_WIN32) || defined(_WIN64)
#include <sprokit/pipeline_util/include-paths.h>
#endif

#include <sprokit/pipeline_util/pipe_declaration_types.h>
#include <sprokit/pipeline_util/pipe_parser.h>
#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline/pipeline.h>

#include <vital/config/config_block.h>
#include <vital/util/tokenize.h>
#include <vital/util/string.h>

#include <vital/algorithm_plugin_manager_paths.h> //+ maybe rename later

#include <kwiversys/SystemTools.hxx>

#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace sprokit {

typedef kwiversys::SystemTools ST;

namespace {

static std::string const default_include_dirs = std::string( DEFAULT_PIPE_INCLUDE_PATHS );
static std::string const sprokit_include_envvar = std::string( "SPROKIT_PIPE_INCLUDE_PATH" );
static std::string const split_str = "=";

}


// ==================================================================
pipeline_builder
::pipeline_builder()
  : m_logger( kwiver::vital::get_logger( "sprokit.pipeline_builder" ) )
  , m_blocks()
{
  // extract search paths from env and default
  process_env();
}


// ------------------------------------------------------------------
void
pipeline_builder
::load_pipeline( std::istream& istr, kwiver::vital::path_t const& def_file )
{
  sprokit::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  // process the input stream
  m_blocks = the_parser.parse_pipeline( istr, def_file );
}


// ------------------------------------------------------------------
void
pipeline_builder
::load_pipeline( kwiver::vital::path_t const& def_file )
{
  sprokit::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  std::ifstream input( def_file );
  if ( ! input )
  {
    VITAL_THROW( sprokit::file_no_exist_exception, def_file );
  }

  // process the input stream
  m_blocks = the_parser.parse_pipeline( input, def_file );
}

// ----------------------------------------------------------------------------
void
pipeline_builder
::load_cluster(std::istream& istr, kwiver::vital::path_t const& def_file)
{
  sprokit::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  // process the input stream
  m_cluster_blocks = the_parser.parse_cluster( istr, def_file );
}


// ----------------------------------------------------------------------------
void
pipeline_builder
::load_cluster( kwiver::vital::path_t const& def_file )
{
  sprokit::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  std::ifstream input( def_file );
  if ( ! input )
  {
    VITAL_THROW( sprokit::file_no_exist_exception, def_file );
  }

  // process the input stream
  m_cluster_blocks = the_parser.parse_cluster( input, def_file );
}


// ------------------------------------------------------------------
void
pipeline_builder
::load_supplement( kwiver::vital::path_t const& path)
{
  sprokit::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  std::ifstream input( path );
  if ( ! input )
  {
    VITAL_THROW( sprokit::file_no_exist_exception, path );
  }

  // process the input stream
  sprokit::pipe_blocks const supplement = the_parser.parse_pipeline( input, path );

  m_blocks.insert(m_blocks.end(), supplement.begin(), supplement.end());
}


// ------------------------------------------------------------------
void
pipeline_builder
::add_setting( std::string const& setting )
{
  static auto command_line_src = std::make_shared< std::string >( "Command Line" );
  size_t const split_pos = setting.find(split_str);

  if (split_pos == std::string::npos)
  {
    std::string const reason = "Error: The setting on the command line \'" + setting + "\' does not contain "
                               "the \'" + split_str + "\' string which separates the key from the value";

    throw std::runtime_error(reason);
  }

  kwiver::vital::config_block_key_t setting_key = setting.substr(0, split_pos);
  kwiver::vital::config_block_value_t setting_value = setting.substr(split_pos + split_str.size());

  kwiver::vital::config_block_keys_t keys;

  kwiver::vital::tokenize( setting_key, keys,
                 kwiver::vital::config_block::block_sep(),
                 kwiver::vital::TokenizeTrimEmpty );

  if (keys.size() < 2)
  {
    std::string const reason = "Error: The key portion of setting \'" + setting + "\' does not contain "
                               "at least two keys in its keypath which is invalid. (e.g. must be at least a:b)";

    throw std::runtime_error(reason);
  }

  sprokit::config_value_t value;
  value.key_path.push_back(keys.back());
  value.value = setting_value;
  value.loc = ::kwiver::vital::source_location( command_line_src, 1 );
  keys.pop_back();

  sprokit::config_pipe_block block;
  block.key = keys;
  block.values.push_back(value);
  block.loc = ::kwiver::vital::source_location( command_line_src, 1 );

  // Add to pipe blocks
  m_blocks.push_back(block);
}


// ------------------------------------------------------------------
void
pipeline_builder
::add_search_path( kwiver::vital::config_path_t const& file_path )
{
  m_search_path.push_back( file_path );
  LOG_DEBUG( m_logger, "Adding \"" << file_path << "\" to search path" );
}


// ------------------------------------------------------------------
void
pipeline_builder
::add_search_path( kwiver::vital::config_path_list_t const& file_path )
{
  if ( file_path.size() > 0 )
  {
    m_search_path.insert( m_search_path.end(),
                          file_path.begin(), file_path.end() );

    LOG_DEBUG( m_logger, "Adding \"" << kwiver::vital::join( file_path, ", " )
               << "\" to search path" );
  }
}


// ------------------------------------------------------------------
sprokit::pipeline_t
pipeline_builder
::pipeline() const
{
  return sprokit::bake_pipe_blocks(m_blocks);
}


// ----------------------------------------------------------------------------
sprokit::cluster_info_t
pipeline_builder
::cluster_info() const
{
  return sprokit::bake_cluster_blocks( m_cluster_blocks );
}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
pipeline_builder
::config() const
{
  return sprokit::extract_configuration(m_blocks);
}


// ------------------------------------------------------------------
sprokit::pipe_blocks
pipeline_builder
::pipeline_blocks() const
{
  return m_blocks;
}


// ------------------------------------------------------------------
sprokit::cluster_blocks
pipeline_builder
::cluster_blocks() const
{
  return m_cluster_blocks;
}


// ----------------------------------------------------------------------------
void
pipeline_builder
::process_env()
{
  // Add path from the environment
  kwiver::vital::path_list_t path_list;
  kwiversys::SystemTools::GetPath( path_list, sprokit_include_envvar.c_str() );

  // Add the default search path
  ST::Split( default_include_dirs, path_list, PATH_SEPARATOR_CHAR );

  if ( ! path_list.empty() )
  {
    add_search_path( path_list );
  }
}


} // end namespace
