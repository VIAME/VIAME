/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "pipe_check.h"

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/config_block_io.h>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/plugin_factory.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/algorithm_framework/plugin/plugin_manager_internal.h>

#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/process.h>
#include <viame/pipeline_framework/pipeline_builder.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
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
namespace fs = std::filesystem;
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
/// One `<something>:type` key and the implementation it names.
struct algo_selection
{
  std::string impl;
  bool resolved = false;
};

// ----------------------------------------------------------------------------
struct process_result
{
  std::string type;
  std::map< std::string, algo_selection > algos;
};

// ----------------------------------------------------------------------------
struct file_result
{
  std::string status;   // ok or error
  std::string message;
  std::map< std::string, process_result > processes;

  // Algorithm selections of a .conf file, which has no processes to hang
  // them off of.
  std::map< std::string, algo_selection > algos;
};

// ----------------------------------------------------------------------------
/// Every implementation name the registry knows, whatever its interface.
///
/// A pipeline names an implementation without naming its interface, so this
/// is the set a `:type` value has to land in to be resolvable.
std::set< std::string >
registered_implementations()
{
  std::set< std::string > names;

  auto& pm = kv::plugin_manager_internal::instance();

  for( auto const& interface_entry : pm.plugin_map() )
  {
    for( auto const& fact : interface_entry.second )
    {
      std::string category;
      fact->get_attribute( kvpf::PLUGIN_CATEGORY, category );

      if( category == kvpf::PROCESS_CATEGORY ||
          category == kvpf::CLUSTER_CATEGORY ||
          category == kvpf::APPLET_CATEGORY )
      {
        continue;
      }

      std::string name;
      if( fact->get_attribute( kvpf::PLUGIN_NAME, name ) && !name.empty() )
      {
        names.insert( name );
      }
    }
  }

  return names;
}

// ----------------------------------------------------------------------------
bool
ends_with( std::string const& value, std::string const& suffix )
{
  return value.size() >= suffix.size() &&
         value.compare( value.size() - suffix.size(),
                        suffix.size(), suffix ) == 0;
}

// ----------------------------------------------------------------------------
/// Record every `:type` key of \p config under \p prefix.
///
/// The prefix is the process name and is stripped from the reported key, so
/// the same algorithm reads the same whatever the process is called.
void
collect_algo_selections( kv::config_block_sptr const& config,
                         std::string const& prefix,
                         std::set< std::string > const& implementations,
                         std::map< std::string, algo_selection >& out )
{
  static const std::string type_suffix = ":type";

  for( auto const& key : config->available_values() )
  {
    if( !ends_with( key, type_suffix ) )
    {
      continue;
    }

    if( !prefix.empty() &&
        key.compare( 0, prefix.size(), prefix ) != 0 )
    {
      continue;
    }

    std::string const local = prefix.empty() ? key : key.substr( prefix.size() );

    algo_selection selection;
    selection.impl = config->get_value< std::string >( key, "" );

    if( selection.impl.empty() )
    {
      continue;
    }

    selection.resolved =
      implementations.find( selection.impl ) != implementations.end();

    out[ local ] = selection;
  }
}

// ----------------------------------------------------------------------------
file_result
check_pipe_file( fs::path const& path,
                 std::vector< std::string > const& search_paths,
                 std::set< std::string > const& implementations )
{
  file_result result;

  try
  {
    sprokit::pipeline_builder builder;

    for( auto const& dir : search_paths )
    {
      builder.add_search_path( dir );
    }

    builder.load_pipeline( path.string() );

    auto const config = builder.config();
    auto const pipe = builder.pipeline();

    if( !pipe )
    {
      result.status = "error";
      result.message = "pipeline could not be built";
      return result;
    }

    for( auto const& name : pipe->process_names() )
    {
      process_result process;
      process.type = pipe->process_by_name( name )->type();

      collect_algo_selections( config, name + ":", implementations,
                               process.algos );

      result.processes[ name ] = process;
    }

    result.status = "ok";
  }
  catch( std::exception const& e )
  {
    result.status = "error";
    result.message = e.what();
  }

  return result;
}

// ----------------------------------------------------------------------------
file_result
check_conf_file( fs::path const& path,
                 std::vector< std::string > const& search_paths,
                 std::set< std::string > const& implementations )
{
  file_result result;

  try
  {
    kv::config_path_list_t paths( search_paths.begin(), search_paths.end() );
    auto const config = kv::read_config_file( path.string(), paths );

    if( !config )
    {
      result.status = "error";
      result.message = "configuration could not be read";
      return result;
    }

    collect_algo_selections( config, "", implementations, result.algos );
    result.status = "ok";
  }
  catch( std::exception const& e )
  {
    result.status = "error";
    result.message = e.what();
  }

  return result;
}

// ----------------------------------------------------------------------------
void
write_algos( std::ostream& os,
             std::map< std::string, algo_selection > const& algos,
             unsigned level )
{
  os << "{";

  if( algos.empty() )
  {
    os << "}";
    return;
  }

  bool first = true;
  for( auto const& item : algos )
  {
    os << ( first ? "\n" : ",\n" ) << indent( level + 1 )
       << quote( item.first ) << ": {\n"
       << indent( level + 2 ) << "\"impl\": " << quote( item.second.impl )
       << ",\n" << indent( level + 2 ) << "\"resolved\": "
       << ( item.second.resolved ? "true" : "false" )
       << "\n" << indent( level + 1 ) << "}";
    first = false;
  }

  os << "\n" << indent( level ) << "}";
}

// ----------------------------------------------------------------------------
void
write_result( std::ostream& os, file_result const& result, unsigned level )
{
  os << "{\n" << indent( level + 1 ) << "\"status\": "
     << quote( result.status ) << ",\n"
     << indent( level + 1 ) << "\"message\": " << quote( result.message )
     << ",\n" << indent( level + 1 ) << "\"processes\": {";

  if( !result.processes.empty() )
  {
    bool first = true;
    for( auto const& item : result.processes )
    {
      os << ( first ? "\n" : ",\n" ) << indent( level + 2 )
         << quote( item.first ) << ": {\n"
         << indent( level + 3 ) << "\"type\": " << quote( item.second.type )
         << ",\n" << indent( level + 3 ) << "\"algos\": ";
      write_algos( os, item.second.algos, level + 3 );
      os << "\n" << indent( level + 2 ) << "}";
      first = false;
    }
    os << "\n" << indent( level + 1 );
  }

  os << "},\n" << indent( level + 1 ) << "\"algos\": ";
  write_algos( os, result.algos, level + 1 );
  os << "\n" << indent( level ) << "}";
}

// ----------------------------------------------------------------------------
/// Directories `--all` walks, relative to the install root.
std::vector< std::string > const&
default_roots()
{
  static std::vector< std::string > const roots = {
    "configs/pipelines",
    "configs/add-ons",
    "examples",
  };

  return roots;
}

// ----------------------------------------------------------------------------
/// The install prefix, which is where the add-on pipelines have been unpacked.
std::string
install_root()
{
  char const* const env = std::getenv( "VIAME_INSTALL" );
  return env ? std::string( env ) : std::string();
}

// ----------------------------------------------------------------------------
void
collect_files( fs::path const& dir, std::vector< fs::path >& files )
{
  std::error_code ec;

  if( !fs::is_directory( dir, ec ) )
  {
    return;
  }

  for( auto const& entry : fs::recursive_directory_iterator(
         dir, fs::directory_options::skip_permission_denied, ec ) )
  {
    if( !entry.is_regular_file( ec ) )
    {
      continue;
    }

    auto const ext = entry.path().extension().string();

    if( ext == ".pipe" || ext == ".conf" )
    {
      files.push_back( entry.path() );
    }
  }
}

} // namespace

// ----------------------------------------------------------------------------
void
pipe_check_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "a,all", "Check every pipeline and configuration file under the "
      "install root",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "i,input", "A single .pipe/.conf file or a directory to walk",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "r,root", "Install root to resolve --all and report paths against "
      "(default: $VIAME_INSTALL)",
      ::cxxopts::value< std::string >()->default_value( "" ), "dir" )
    ( "json", "Emit JSON; the only supported format, accepted for symmetry "
      "with the other tools",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "o,output", "Output .json file (default: stdout)",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ;
}

// ----------------------------------------------------------------------------
int
pipe_check_applet
::run()
{
  auto logger = kv::get_logger( "viame.tools.pipe_check" );
  auto& cmd_args = command_args();

  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame pipe-check [options]\n\n"
              << "Bake pipeline files the way the runner does and report "
                 "what each\n"
              << "process resolves to. With --all, walk every pipeline and "
                 "config\n"
              << "file shipped in the install.\n"
              << m_cmd_options->help()
              << "\nExamples:\n"
              << "  viame pipe-check --all --json > pipes.json\n"
              << "  viame pipe-check -i detector.pipe\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  bool const opt_all = cmd_args[ "all" ].as< bool >();
  std::string const opt_input = cmd_args[ "input" ].as< std::string >();
  std::string const opt_output = cmd_args[ "output" ].as< std::string >();
  std::string opt_root = cmd_args[ "root" ].as< std::string >();

  if( opt_root.empty() )
  {
    opt_root = install_root();
  }

  if( !opt_all && opt_input.empty() )
  {
    LOG_ERROR( logger, "Nothing to check: pass --all or --input." );
    return EXIT_FAILURE;
  }

  if( opt_all && opt_root.empty() )
  {
    LOG_ERROR( logger, "--all needs an install root: pass --root or source "
                       "setup_viame.sh." );
    return EXIT_FAILURE;
  }

  kv::plugin_manager::instance().load_all_plugins();
  auto const implementations = registered_implementations();

  fs::path const root( opt_root );
  std::vector< fs::path > files;

  if( opt_all )
  {
    for( auto const& sub : default_roots() )
    {
      collect_files( root / sub, files );
    }
  }

  if( !opt_input.empty() )
  {
    fs::path const input( opt_input );
    std::error_code ec;

    if( fs::is_directory( input, ec ) )
    {
      collect_files( input, files );
    }
    else if( fs::is_regular_file( input, ec ) )
    {
      files.push_back( input );
    }
    else
    {
      LOG_ERROR( logger, "Input path does not exist: " << opt_input );
      return EXIT_FAILURE;
    }
  }

  // The runner resolves includes against the installed pipeline directory
  std::vector< std::string > search_paths;

  if( !opt_root.empty() )
  {
    search_paths.push_back( ( root / "configs" / "pipelines" ).string() );
    search_paths.push_back( ( root / "configs" ).string() );
  }

  std::map< std::string, file_result > results;

  for( auto const& file : files )
  {
    std::error_code ec;
    auto relative = fs::relative( file, root, ec );
    std::string const key =
      ( ec || relative.empty() || relative.string().rfind( "..", 0 ) == 0 )
      ? file.string() : relative.string();

    // Keep the file's own directory ahead of the install so that a pipeline
    // checked out of tree still finds its neighbours
    auto file_search_paths = search_paths;
    file_search_paths.insert( file_search_paths.begin(),
                              file.parent_path().string() );

    results[ key ] = ( file.extension() == ".conf" )
      ? check_conf_file( file, file_search_paths, implementations )
      : check_pipe_file( file, file_search_paths, implementations );
  }

  std::ostringstream os;
  os << "{";

  bool first = true;
  for( auto const& item : results )
  {
    os << ( first ? "\n" : ",\n" ) << indent( 1 ) << quote( item.first )
       << ": ";
    write_result( os, item.second, 1 );
    first = false;
  }

  os << ( first ? "}\n" : "\n}\n" );

  if( opt_output.empty() )
  {
    std::cout << os.str();
  }
  else
  {
    std::ofstream file( opt_output );

    if( !file.is_open() )
    {
      LOG_ERROR( logger, "Could not open output file: " << opt_output );
      return EXIT_FAILURE;
    }

    file << os.str();
  }

  return EXIT_SUCCESS;
}

} // namespace tools
} // namespace viame
