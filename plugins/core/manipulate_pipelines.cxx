/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "manipulate_pipelines.h"
#include "utilities_file.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

namespace viame {

// =============================================================================
std::string detect_marker_indent(
    const std::string& template_content,
    const std::string& marker )
{
  std::size_t pos = template_content.find( marker );

  if( pos == std::string::npos )
  {
    return "";
  }

  // Scan backwards to the preceding newline or string start
  std::size_t line_start = 0;

  if( pos > 0 )
  {
    std::size_t nl = template_content.rfind( '\n', pos - 1 );

    if( nl != std::string::npos )
    {
      line_start = nl + 1;
    }
  }

  // Extract whitespace characters between line start and marker position
  std::string indent;

  for( std::size_t i = line_start; i < pos; ++i )
  {
    char c = template_content[ i ];

    if( c == ' ' || c == '\t' )
    {
      indent += c;
    }
    else
    {
      break;
    }
  }

  return indent;
}

// =============================================================================
// Internal helper: a single config entry with its prefix and parameter name
namespace {

struct pipe_entry
{
  std::string prefix;
  std::string param_name;
  std::string value;
  bool is_file;
};

} // anonymous namespace

// =============================================================================
std::string format_output_as_pipe_blocks(
    const std::map< std::string, std::string >& config_entries,
    const std::set< std::string >& copied_filenames,
    const std::string& base_indent )
{
  if( config_entries.empty() )
  {
    return "";
  }

  // 1. Prefix each config key with "detector:" and split at last ':'
  std::vector< pipe_entry > entries;

  for( const auto& pair : config_entries )
  {
    pipe_entry e;
    std::string full_key = "detector:" + pair.first;

    // Split at the last ':'
    std::size_t last_colon = full_key.rfind( ':' );

    if( last_colon != std::string::npos && last_colon > 0 )
    {
      e.prefix = full_key.substr( 0, last_colon );
      e.param_name = full_key.substr( last_colon + 1 );
    }
    else
    {
      e.prefix = "";
      e.param_name = full_key;
    }

    e.value = pair.second;
    e.is_file = copied_filenames.count( pair.second ) > 0;
    entries.push_back( e );
  }

  // 2. Group entries by prefix
  std::map< std::string, std::vector< pipe_entry > > groups;

  for( const auto& e : entries )
  {
    groups[ e.prefix ].push_back( e );
  }

  // 3. Build output
  std::ostringstream out;
  bool first_group = true;
  const std::size_t align_col = 47;

  for( const auto& group : groups )
  {
    const std::string& prefix = group.first;
    const std::vector< pipe_entry >& group_entries = group.second;

    // Check if any entry in this group needs relativepath
    bool any_file = false;
    for( const auto& e : group_entries )
    {
      if( e.is_file )
      {
        any_file = true;
        break;
      }
    }

    bool use_block = ( group_entries.size() > 1 ) || any_file;

    // Blank line between groups (except before first)
    if( !first_group )
    {
      out << "\n" << base_indent << "\n";
    }

    if( !use_block )
    {
      // Single inline entry: :<prefix>:<param>  <padded_value>
      const pipe_entry& e = group_entries[0];
      std::string line_content = ":" + prefix + ":" + e.param_name;

      // First line has no leading indent (template provides it)
      if( first_group )
      {
        // Pad to align column
        std::size_t pad = ( line_content.size() < align_col )
            ? ( align_col - line_content.size() ) : 1;
        out << line_content << std::string( pad, ' ' ) << e.value;
      }
      else
      {
        std::string full_line = base_indent + line_content;
        std::size_t pad = ( full_line.size() < align_col )
            ? ( align_col - full_line.size() ) : 1;
        out << base_indent << line_content << std::string( pad, ' ' ) << e.value;
      }
    }
    else
    {
      // Block format
      std::string block_indent = base_indent + "  ";

      if( first_group )
      {
        out << "block " << prefix;
      }
      else
      {
        out << base_indent << "block " << prefix;
      }

      for( const auto& e : group_entries )
      {
        std::string line_content;

        if( e.is_file )
        {
          line_content = "relativepath " + e.param_name + " =";
        }
        else
        {
          line_content = ":" + e.param_name;
        }

        std::string full_line = block_indent + line_content;
        std::size_t pad = ( full_line.size() < align_col )
            ? ( align_col - full_line.size() ) : 1;

        out << "\n" << block_indent << line_content
            << std::string( pad, ' ' ) << e.value;
      }

      out << "\n" << base_indent << "endblock";
    }

    first_group = false;
  }

  return out.str();
}

// =============================================================================
std::string generate_detector_impl_replacement(
    const std::map< std::string, std::string >& output_map,
    const std::string& pipeline_template )
{
  if( pipeline_template.empty() || !does_file_exist( pipeline_template )
      || !file_contains_string( pipeline_template, "[-DETECTOR-IMPL-]" ) )
  {
    return "";
  }

  // Separate config entries from file copies
  std::map< std::string, std::string > config_entries;
  std::set< std::string > copied_filenames;

  for( const auto& pair : output_map )
  {
    if( !pair.second.empty() && does_file_exist( pair.second ) )
    {
      copied_filenames.insert( pair.first );
    }
    else
    {
      config_entries[ pair.first ] = pair.second;
    }
  }

  // Read template and detect indent
  std::ifstream tfile( pipeline_template );
  std::string tcontent( ( std::istreambuf_iterator< char >( tfile ) ),
                          std::istreambuf_iterator< char >() );
  tfile.close();

  std::string indent = detect_marker_indent( tcontent, "[-DETECTOR-IMPL-]" );

  return format_output_as_pipe_blocks( config_entries, copied_filenames, indent );
}

} // end namespace viame
