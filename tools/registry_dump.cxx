/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "registry_dump.h"

#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/plugin_management/plugin_factory.h>
#include <vital/plugin_management/plugin_manager.h>
#include <vital/plugin_management/plugin_manager_internal.h>

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace viame {
namespace tools {

namespace {

namespace kv = kwiver::vital;
typedef kv::plugin_factory kvpf;

// ----------------------------------------------------------------------------
std::string
json_escape( std::string const& input )
{
  std::string output;
  output.reserve( input.size() + 16 );

  for( char c : input )
  {
    switch( c )
    {
      case '"':  output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\b': output += "\\b";  break;
      case '\f': output += "\\f";  break;
      case '\n': output += "\\n";  break;
      case '\r': output += "\\r";  break;
      case '\t': output += "\\t";  break;
      default:
        if( static_cast< unsigned char >( c ) < 0x20 )
        {
          char buf[ 8 ];
          snprintf( buf, sizeof( buf ), "\\u%04x",
                    static_cast< unsigned char >( c ) );
          output += buf;
        }
        else
        {
          output += c;
        }
        break;
    }
  }

  return output;
}

// ----------------------------------------------------------------------------
/// Sort the implementation list an algorithm description ends with.
///
/// The list is built in plugin registration order, which python module import
/// order perturbs from run to run. Sorting it makes the dump reproducible
/// without dropping what the description says.
std::string
normalize_description( std::string const& description )
{
  static const std::string marker = "Must be one of the following options:";
  static const std::string separator = "\n\t- ";

  auto const marker_pos = description.find( marker );

  if( marker_pos == std::string::npos )
  {
    return description;
  }

  auto const list_pos = marker_pos + marker.size();
  std::string const head = description.substr( 0, list_pos );
  std::string const list = description.substr( list_pos );

  if( list.compare( 0, separator.size(), separator ) != 0 )
  {
    return description;
  }

  std::vector< std::string > options;
  std::string::size_type pos = separator.size();

  while( pos <= list.size() )
  {
    auto const next = list.find( separator, pos );
    auto const end = ( next == std::string::npos ) ? list.size() : next;
    options.push_back( list.substr( pos, end - pos ) );

    if( next == std::string::npos )
    {
      break;
    }

    pos = next + separator.size();
  }

  std::sort( options.begin(), options.end() );

  std::string result = head;
  for( auto const& option : options )
  {
    result += separator + option;
  }

  return result;
}

// ----------------------------------------------------------------------------
std::string
quote( std::string const& input )
{
  return "\"" + json_escape( input ) + "\"";
}

// ----------------------------------------------------------------------------
std::string
indent( unsigned level )
{
  return std::string( level * 2, ' ' );
}

// ----------------------------------------------------------------------------
/// One configuration key of an algorithm, process or cluster.
struct config_entry
{
  std::string value;
  std::string description;
  bool tunable = false;
  bool has_tunable = false;
};

typedef std::map< std::string, config_entry > config_map_t;

// ----------------------------------------------------------------------------
/// One port of a process or cluster.
struct port_entry
{
  std::string type;
  std::string description;
  std::vector< std::string > flags;
};

typedef std::map< std::string, port_entry > port_map_t;

// ----------------------------------------------------------------------------
/// A registered name and everything the dump records about it.
struct registered_entry
{
  std::string description;
  config_map_t config;
  port_map_t input_ports;
  port_map_t output_ports;

  // Set when the name is registered but could not be introspected. The
  // baseline still records the name so that its disappearance is caught.
  std::string error;
};

typedef std::map< std::string, registered_entry > entry_map_t;

// ----------------------------------------------------------------------------
void
write_config( std::ostream& os, config_map_t const& config, unsigned level,
              bool descriptions )
{
  os << "{";

  if( config.empty() )
  {
    os << "}";
    return;
  }

  bool first = true;
  for( auto const& item : config )
  {
    os << ( first ? "\n" : ",\n" ) << indent( level + 1 )
       << quote( item.first ) << ": {\n"
       << indent( level + 2 ) << "\"default\": "
       << quote( item.second.value );

    if( descriptions )
    {
      os << ",\n" << indent( level + 2 ) << "\"description\": "
         << quote( normalize_description( item.second.description ) );
    }

    if( item.second.has_tunable )
    {
      os << ",\n" << indent( level + 2 ) << "\"tunable\": "
         << ( item.second.tunable ? "true" : "false" );
    }

    os << "\n" << indent( level + 1 ) << "}";
    first = false;
  }

  os << "\n" << indent( level ) << "}";
}

// ----------------------------------------------------------------------------
void
write_ports( std::ostream& os, port_map_t const& ports, unsigned level,
             bool descriptions )
{
  os << "{";

  if( ports.empty() )
  {
    os << "}";
    return;
  }

  bool first = true;
  for( auto const& item : ports )
  {
    os << ( first ? "\n" : ",\n" ) << indent( level + 1 )
       << quote( item.first ) << ": {\n"
       << indent( level + 2 ) << "\"type\": " << quote( item.second.type )
       << ",\n" << indent( level + 2 ) << "\"flags\": [";

    bool first_flag = true;
    for( auto const& flag : item.second.flags )
    {
      os << ( first_flag ? "" : ", " ) << quote( flag );
      first_flag = false;
    }
    os << "]";

    if( descriptions )
    {
      os << ",\n" << indent( level + 2 ) << "\"description\": "
         << quote( item.second.description );
    }

    os << "\n" << indent( level + 1 ) << "}";
    first = false;
  }

  os << "\n" << indent( level ) << "}";
}

// ----------------------------------------------------------------------------
/// Write one named group of entries, emitting only the fields it carries.
void
write_entries( std::ostream& os, entry_map_t const& entries, unsigned level,
               bool descriptions, bool with_ports )
{
  os << "{";

  if( entries.empty() )
  {
    os << "}";
    return;
  }

  bool first = true;
  for( auto const& item : entries )
  {
    auto const& entry = item.second;

    os << ( first ? "\n" : ",\n" ) << indent( level + 1 )
       << quote( item.first ) << ": {\n";

    if( descriptions )
    {
      os << indent( level + 2 ) << "\"description\": "
         << quote( entry.description ) << ",\n";
    }

    if( !entry.error.empty() )
    {
      os << indent( level + 2 ) << "\"error\": " << quote( entry.error )
         << ",\n";
    }

    os << indent( level + 2 ) << "\"config\": ";
    write_config( os, entry.config, level + 2, descriptions );

    if( with_ports )
    {
      os << ",\n" << indent( level + 2 ) << "\"input_ports\": ";
      write_ports( os, entry.input_ports, level + 2, descriptions );
      os << ",\n" << indent( level + 2 ) << "\"output_ports\": ";
      write_ports( os, entry.output_ports, level + 2, descriptions );
    }

    os << "\n" << indent( level + 1 ) << "}";
    first = false;
  }

  os << "\n" << indent( level ) << "}";
}

// ----------------------------------------------------------------------------
std::string
attribute( kv::plugin_factory_handle_t const& fact, std::string const& key )
{
  std::string value;
  fact->get_attribute( key, value );
  return value;
}

// ----------------------------------------------------------------------------
/// Fill \p config from an algorithm factory's declared default configuration.
void
collect_algorithm_config( kv::plugin_factory_handle_t const& fact,
                          registered_entry& entry )
{
  auto block = kv::config_block::empty_config();

  try
  {
    fact->get_default_config( *block );
  }
  catch( std::exception const& e )
  {
    entry.error = e.what();
    return;
  }

  for( auto const& key : block->available_values() )
  {
    config_entry item;
    item.value = block->get_value< std::string >( key, "" );
    item.description = block->get_description( key );
    entry.config[ key ] = item;
  }
}

// ----------------------------------------------------------------------------
/// Instantiate a process and record its configuration and ports.
void
collect_process_details( std::string const& type, registered_entry& entry )
{
  sprokit::process_t proc;

  try
  {
    proc = sprokit::create_process( type, "registry_dump_probe" );
  }
  catch( std::exception const& e )
  {
    entry.error = e.what();
    return;
  }

  if( !proc )
  {
    entry.error = "process could not be created";
    return;
  }

  try
  {
    for( auto const& key : proc->available_config() )
    {
      config_entry item;

      try
      {
        auto const info = proc->config_info( key );
        item.value = info->def;
        item.description = info->description;
        item.tunable = info->tunable;
        item.has_tunable = true;
      }
      catch( std::exception const& e )
      {
        item.description = std::string( "unavailable: " ) + e.what();
      }

      entry.config[ key ] = item;
    }

    for( auto const& port : proc->input_ports() )
    {
      auto const info = proc->input_port_info( port );
      port_entry item;
      item.type = info->type;
      item.description = info->description;
      item.flags.assign( info->flags.begin(), info->flags.end() );
      entry.input_ports[ port ] = item;
    }

    for( auto const& port : proc->output_ports() )
    {
      auto const info = proc->output_port_info( port );
      port_entry item;
      item.type = info->type;
      item.description = info->description;
      item.flags.assign( info->flags.begin(), info->flags.end() );
      entry.output_ports[ port ] = item;
    }
  }
  catch( std::exception const& e )
  {
    entry.error = e.what();
  }
}

// ----------------------------------------------------------------------------
/// The python packages the runtime is told to import, in sorted order.
std::vector< std::string >
python_modules()
{
  std::set< std::string > modules;

  char const* const env = std::getenv( "SPROKIT_PYTHON_MODULES" );

  if( env )
  {
    std::istringstream ss( env );
    std::string module;

    while( std::getline( ss, module, ':' ) )
    {
      if( !module.empty() )
      {
        modules.insert( module );
      }
    }
  }

  return std::vector< std::string >( modules.begin(), modules.end() );
}

} // namespace

// ----------------------------------------------------------------------------
void
registry_dump_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "json", "Emit JSON; the only supported format, accepted for symmetry "
      "with the other tools",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "o,output", "Output .json file (default: stdout)",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "no-descriptions", "Exclude descriptions from the output",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ;
}

// ----------------------------------------------------------------------------
int
registry_dump_applet
::run()
{
  auto& cmd_args = command_args();

  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame registry-dump [options]\n\n"
              << "Load every plugin this build provides and write what they\n"
              << "register as JSON: algorithm interfaces and their "
                 "implementations,\n"
              << "sprokit processes and clusters with their config keys and "
                 "ports,\n"
              << "schedulers, and applets.\n"
              << m_cmd_options->help()
              << std::endl;
    return EXIT_SUCCESS;
  }

  bool const descriptions = !cmd_args[ "no-descriptions" ].as< bool >();
  std::string const output_file = cmd_args[ "output" ].as< std::string >();

  auto& pm = kv::plugin_manager_internal::instance();
  pm.load_all_plugins();

  std::map< std::string, entry_map_t > algorithms;
  entry_map_t processes;
  entry_map_t clusters;
  entry_map_t schedulers;
  entry_map_t applets;

  for( auto const& interface_entry : pm.plugin_map() )
  {
    std::string const& interface_name = interface_entry.first;

    for( auto const& fact : interface_entry.second )
    {
      std::string const name = attribute( fact, kvpf::PLUGIN_NAME );

      if( name.empty() )
      {
        continue;
      }

      std::string const category = attribute( fact, kvpf::PLUGIN_CATEGORY );

      registered_entry entry;
      entry.description = attribute( fact, kvpf::PLUGIN_DESCRIPTION );

      if( category == kvpf::PROCESS_CATEGORY )
      {
        collect_process_details( name, entry );
        processes[ name ] = entry;
      }
      else if( category == kvpf::CLUSTER_CATEGORY )
      {
        collect_process_details( name, entry );
        clusters[ name ] = entry;
      }
      else if( category == kvpf::APPLET_CATEGORY )
      {
        applets[ name ] = entry;
      }
      else if( category == "scheduler" )
      {
        schedulers[ name ] = entry;
      }
      else
      {
        collect_algorithm_config( fact, entry );
        algorithms[ interface_name ][ name ] = entry;
      }
    }
  }

  std::ostringstream os;

  os << "{\n" << indent( 1 ) << "\"aliases\": {},\n"
     << indent( 1 ) << "\"algorithms\": {";

  bool first_interface = true;
  for( auto const& item : algorithms )
  {
    os << ( first_interface ? "\n" : ",\n" ) << indent( 2 )
       << quote( item.first ) << ": ";
    write_entries( os, item.second, 2, descriptions, false );
    first_interface = false;
  }
  os << ( algorithms.empty() ? "}" : "\n" + indent( 1 ) + "}" ) << ",\n";

  os << indent( 1 ) << "\"applets\": ";
  write_entries( os, applets, 1, descriptions, false );
  os << ",\n";

  os << indent( 1 ) << "\"clusters\": ";
  write_entries( os, clusters, 1, descriptions, true );
  os << ",\n";

  os << indent( 1 ) << "\"processes\": ";
  write_entries( os, processes, 1, descriptions, true );
  os << ",\n";

  os << indent( 1 ) << "\"python_modules\": [";
  bool first_module = true;
  for( auto const& module : python_modules() )
  {
    os << ( first_module ? "\n" : ",\n" ) << indent( 2 ) << quote( module );
    first_module = false;
  }
  os << ( first_module ? "]" : "\n" + indent( 1 ) + "]" ) << ",\n";

  os << indent( 1 ) << "\"schedulers\": ";
  write_entries( os, schedulers, 1, descriptions, false );
  os << "\n}\n";

  if( output_file.empty() )
  {
    std::cout << os.str();
  }
  else
  {
    std::ofstream file( output_file );

    if( !file.is_open() )
    {
      auto logger = kv::get_logger( "viame.tools.registry_dump" );
      LOG_ERROR( logger, "Could not open output file: " << output_file );
      return EXIT_FAILURE;
    }

    file << os.str();
  }

  return EXIT_SUCCESS;
}

} // namespace tools
} // namespace viame
