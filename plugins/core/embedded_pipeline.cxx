/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "embedded_pipeline.h"

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/processes/adapters/embedded_pipeline.h>

#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace viame {
namespace {

using processes_t = std::vector< std::pair< std::string, std::string > >;
using names_t = std::set< std::string >;

std::vector< std::string > select_processes(
  processes_t const& processes,
  std::optional< std::vector< std::string > > const& requested,
  names_t const& defaults, std::string const& label )
{
  std::vector< std::string > result;
  names_t available;
  std::string listing;
  for( auto const& process : processes )
  {
    available.insert( process.first );
    if( !listing.empty() ) { listing += ", "; }
    listing += process.first;
    if( !requested && defaults.count( process.second ) )
    {
      result.push_back( process.first );
    }
  }
  if( requested ) { result = *requested; }
  names_t seen;
  bool valid = !result.empty();
  for( auto const& name : result )
  {
    if( !available.count( name ) || !seen.insert( name ).second )
    {
      valid = false;
    }
  }
  if( !valid )
  {
    throw std::invalid_argument( "Select " + label + " process names with " +
                                 label + "=; available: " + listing );
  }
  return result;
}

std::string mapped_port( std::map< std::string, std::string >& ports,
                         sprokit::process::port_addr_t const& address,
                         std::string const& prefix )
{
  auto const key = address.first + "." + address.second;
  return ports.emplace( key, prefix + std::to_string( ports.size() ) ).first->second;
}

} // namespace

embedded_pipeline_description prepare_embedded_pipeline(
  std::string const& filename, embedded_pipeline_options const& options )
{
  auto const path = std::filesystem::absolute( filename );
  sprokit::pipeline_builder builder;
  for( auto const& dir : options.search_paths ) { builder.add_search_path( dir ); }
  builder.load_pipeline( path.string() );

  processes_t processes;
  names_t process_names;
  std::vector< sprokit::connect_pipe_block > edges;
  for( auto const& block : builder.pipeline_blocks() )
  {
    if( auto const* proc = std::get_if< sprokit::process_pipe_block >( &block ) )
    {
      if( !process_names.insert( proc->name ).second )
      {
        throw std::invalid_argument( "Duplicate process: " + proc->name );
      }
      processes.emplace_back( proc->name, proc->type );
    }
    else if( auto const* edge = std::get_if< sprokit::connect_pipe_block >( &block ) )
    {
      edges.push_back( *edge );
    }
  }

  embedded_pipeline_description result;
  result.source_directory = path.parent_path().string();
  result.input_names = select_processes( processes, options.inputs,
    { "video_input", "image_list_reader", "frame_list_input", "input_adapter" },
    "inputs" );
  auto const output_names = select_processes( processes, options.outputs,
    { "detected_object_output", "write_object_track", "image_writer",
      "video_output", "kw_write_homography", "write_track_descriptor", "output_adapter" },
    "outputs" );
  names_t const inputs( result.input_names.begin(), result.input_names.end() );
  names_t const outputs( output_names.begin(), output_names.end() );
  names_t removed = inputs;
  for( auto const& name : outputs )
  {
    if( !removed.insert( name ).second )
    {
      throw std::invalid_argument( "Input and output processes must be distinct" );
    }
  }

  std::string in_name = "viame_memory_input", out_name = "viame_memory_output";
  while( process_names.count( in_name ) ) { in_name += "_"; }
  while( process_names.count( out_name ) ) { out_name += "_"; }
  for( auto& edge : edges )
  {
    if( outputs.count( edge.from.first ) )
    {
      throw std::invalid_argument( "Selected output processes must be sinks" );
    }
    if( inputs.count( edge.to.first ) )
    {
      throw std::invalid_argument( "Selected input processes must be sources" );
    }
    if( inputs.count( edge.from.first ) )
    {
      auto const alias = mapped_port( result.input_ports, edge.from, "input_" );
      edge.from = { in_name, alias };
    }
    if( outputs.count( edge.to.first ) )
    {
      auto const alias = mapped_port( result.output_ports, edge.to, "output_" );
      edge.to = { out_name, alias };
    }
  }
  if( result.input_ports.empty() || result.output_ports.empty() )
  {
    throw std::invalid_argument( "Embedded pipelines need connected input and output ports" );
  }

  std::ostringstream text;
  for( auto const& process : processes )
  {
    if( !removed.count( process.first ) )
    {
      text << "process " << process.first << "\n  :: " << process.second << "\n\n";
    }
  }
  text << "process " << in_name << "\n  :: input_adapter\n\n"
       << "process " << out_name << "\n  :: output_adapter\n\n";
  // Resolve configuration while original source locations are still attached.
  // In particular, paths in included files must not become relative to the
  // caller's working directory or to a generated pipeline file.
  auto const config = builder.config();
  for( auto const& key : config->available_values() )
  {
    auto const split = key.find( ':' );
    auto const root = key.substr( 0, split );
    if( removed.count( root ) ) { continue; }
    auto const value = config->get_value< std::string >( key );
    if( split == std::string::npos || value.find_first_of( "\r\n" ) != std::string::npos )
    {
      throw std::invalid_argument( "Cannot serialize pipeline configuration key: " + key );
    }
    text << "config " << root << "\n  " << key.substr( split + 1 ) << " = " << value << "\n\n";
  }
  for( auto const& edge : edges )
  {
    text << "connect from " << edge.from.first << "." << edge.from.second
         << "\n        to " << edge.to.first << "." << edge.to.second << "\n\n";
  }
  result.pipeline_text = text.str();
  return result;
}

void embedded_pipeline_description::build( kwiver::embedded_pipeline& pipeline ) const
{
  std::istringstream text( pipeline_text );
  pipeline.build_pipeline( text, source_directory );
}

} // namespace viame
