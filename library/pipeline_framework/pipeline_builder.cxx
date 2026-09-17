// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipeline_builder.h"

#include <viame/pipeline_framework/pipe_declaration_types.h>
#include <viame/pipeline_framework/pipe_parser.h>
#include <viame/pipeline_framework/load_pipe_exception.h>
#include <viame/pipeline_framework/pipeline.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/util/tokenize.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/algorithm_framework/kwiver-include-paths.h>

#include <viame/algorithm_framework/util/file_system.h>

#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace viame::pipeline {


namespace {

static std::string const default_include_dirs = std::string( DEFAULT_PIPE_INCLUDE_PATHS );
static std::string const include_envvar =
  std::string( "VIAME_PIPE_INCLUDE_PATH" );
// The name before phase 11, still read with a one-time warning.
static std::string const old_include_envvar =
  std::string( "SPROKIT_PIPE_INCLUDE_PATH" );
static std::string const split_str = "=";
static std::string const path_separator( 1, PATH_SEPARATOR_CHAR );

}

// ==================================================================
pipeline_builder
::pipeline_builder()
  : m_logger( viame::get_logger( "viame.pipeline_builder" ) )
  , m_blocks()
{
  // extract search paths from env and default
  process_env();
}

// ------------------------------------------------------------------
void
pipeline_builder
::load_pipeline( std::istream& istr, viame::path_t const& def_file )
{
  viame::pipeline::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  // process the input stream
  m_blocks = the_parser.parse_pipeline( istr, def_file );
}

// ------------------------------------------------------------------
void
pipeline_builder
::load_pipeline( viame::path_t const& def_file )
{
  viame::pipeline::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  std::ifstream input( def_file );
  if ( ! input )
  {
    VITAL_THROW( viame::pipeline::file_no_exist_exception, def_file );
  }

  // process the input stream
  m_blocks = the_parser.parse_pipeline( input, def_file );
}

// ------------------------------------------------------------------
void
pipeline_builder
::load_supplement( viame::path_t const& path)
{
  viame::pipeline::pipe_parser the_parser;
  the_parser.add_search_path( m_search_path );

  std::ifstream input( path );
  if ( ! input )
  {
    VITAL_THROW( viame::pipeline::file_no_exist_exception, path );
  }

  // process the input stream
  viame::pipeline::pipe_blocks const supplement = the_parser.parse_pipeline( input, path );

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

  viame::config_block_key_t setting_key = setting.substr(0, split_pos);
  viame::config_block_value_t setting_value = setting.substr(split_pos + split_str.size());

  viame::config_block_keys_t keys;

  viame::tokenize( setting_key, keys,
                 viame::config_block::block_sep(),
                 viame::TokenizeTrimEmpty );

  if (keys.size() < 2)
  {
    std::string const reason = "Error: The key portion of setting \'" + setting + "\' does not contain "
                               "at least two keys in its keypath which is invalid. (e.g. must be at least a:b)";

    throw std::runtime_error(reason);
  }

  viame::pipeline::config_value_t value;
  value.key_path.push_back(keys.back());
  value.value = setting_value;
  value.loc = ::viame::source_location( command_line_src, 1 );
  keys.pop_back();

  viame::pipeline::config_pipe_block block;
  block.key = keys;
  block.values.push_back(value);
  block.loc = ::viame::source_location( command_line_src, 1 );

  // Add to pipe blocks
  m_blocks.push_back(block);
}

// ------------------------------------------------------------------
void
pipeline_builder
::add_search_path( viame::config_path_t const& file_path )
{
  m_search_path.push_back( file_path );
  LOG_DEBUG( m_logger, "Adding \"" << file_path << "\" to search path" );
}

// ------------------------------------------------------------------
void
pipeline_builder
::add_search_path( viame::config_path_list_t const& file_path )
{
  if ( file_path.size() > 0 )
  {
    m_search_path.insert( m_search_path.end(),
                          file_path.begin(), file_path.end() );

    LOG_DEBUG( m_logger, "Adding \"" << viame::join( file_path, ", " )
               << "\" to search path" );
  }
}

// ------------------------------------------------------------------
viame::pipeline::pipeline_t
pipeline_builder
::pipeline() const
{
  return viame::pipeline::bake_pipe_blocks(m_blocks);
}

// ------------------------------------------------------------------
viame::config_block_sptr
pipeline_builder
::config() const
{
  return viame::pipeline::extract_configuration(m_blocks);
}

// ------------------------------------------------------------------
viame::pipeline::pipe_blocks
pipeline_builder
::pipeline_blocks() const
{
  return m_blocks;
}

// ----------------------------------------------------------------------------
void
pipeline_builder
::process_env()
{
  // Add path from the environment
  viame::path_list_t path_list;
  viame::environment_path_renamed(
    include_envvar, old_include_envvar, path_list );

  // Add the default search path
  ::viame::tokenize( default_include_dirs, path_list, path_separator,
                             viame::TokenizeTrimEmpty );
  if ( ! path_list.empty() )
  {
    add_search_path( path_list );
  }
}

} // namespace viame::pipeline
