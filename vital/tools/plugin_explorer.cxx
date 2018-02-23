/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
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

#include "explorer_plugin.h"
#include "explorer_context_priv.h"

#include <vital/algorithm_plugin_manager_paths.h> //+ maybe rename later

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/util/demangle.h>
#include <vital/util/get_paths.h>
#include <vital/util/wrap_text_block.h>
#include <vital/logger/logger.h>
#include <vital/algo/algorithm_factory.h>

#include <kwiversys/RegularExpression.hxx>
#include <kwiversys/SystemTools.hxx>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <memory>
#include <map>
#include <algorithm>


/*
TODO

- expand help text to be more like a man page.
- handle processopedia and algo_explorer personalities.

- make it easy to display one factory (e.g. process, algo) Maybe add flag
  for finding impl type by regexp.

- configuration output is different for algos va proc's. These should
  be the same. Maybe some common code would help, although they have
  slightly different contents.

 */

typedef kwiversys::SystemTools ST;

// -----------------------------------------------------------------
/**
 *
 *
 */
class local_manager
  : public kwiver::vital::plugin_manager
{
public:
  local_manager() { }

  kwiver::vital::plugin_loader* loader() { return get_loader(); }

}; // end class local_manager

// -- forward definitions --
static void display_attributes( kwiver::vital::plugin_factory_handle_t const fact );
static void display_by_category( const kwiver::vital::plugin_map_t& plugin_map, const std::string& category );
static kwiver::vital::category_explorer* get_category_handler( const std::string& cat );


//==================================================================
// Define global program data
static kwiver::vital::explorer_context::priv G_context;
static kwiver::vital::explorer_context* G_explorer_context;

static kwiver::vital::logger_handle_t G_logger;

static kwiversys::RegularExpression filter_regex;
static kwiversys::RegularExpression fact_regex;
static kwiversys::RegularExpression type_regex;

#ifndef PLUGIN_EXPLORER_VERSION
  #define PLUGIN_EXPLORER_VERSION "undefined"
#endif

static bool opt_version(false);
static std::string version_string( PLUGIN_EXPLORER_VERSION );

// This program can have different personalities depending on the name
// of the executable.  This is to emulate the useful behaviour of
// older dedicated programs.
enum { prog_default, prog_processopedia, prog_alog_explorer };
static int program_personality( prog_default );

static std::map< const std::string, kwiver::vital::category_explorer *> category_map;

// ==================================================================

static std::string const hidden_prefix = "_";

// Accessor for output stream.
inline std::ostream& pe_out()
{
  return *G_context.m_out_stream;
}


// ------------------------------------------------------------------
/*
 * Functor to print an attribute
 */
struct print_functor
{
  print_functor( std::ostream& str)
    : m_str( str )
  { }

  void operator() ( std::string const& key, std::string const& val ) const
  {
    // Skip canonical names and other attributes that are displayed elsewhere
    if ( ( kwiver::vital::plugin_factory::PLUGIN_NAME != key)
         && ( kwiver::vital::plugin_factory::CONCRETE_TYPE != key )
         && ( kwiver::vital::plugin_factory::INTERFACE_TYPE != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_FILE_NAME != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_VERSION != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME != key )
      )
    {
      std::string value(val);

      size_t pos = value.find( '\004' );
      if ( std::string::npos != pos)
      {
        value.replace( pos, 1, "\"  ::  \"" );
      }

      m_str << "    * " << key << ": \"" << value << "\"" << std::endl;
    }
  }

  // instance data
  std::ostream& m_str;
};


// ------------------------------------------------------------------
std::string
join( const std::vector< std::string >& vec, const char* delim )
{
  std::stringstream res;
  std::copy( vec.begin(), vec.end(), std::ostream_iterator< std::string > ( res, delim ) );

  return res.str();
}


// ------------------------------------------------------------------
void
display_attributes( kwiver::vital::plugin_factory_handle_t const fact )
{
  // See if this factory is selected
  if ( G_context.opt_attr_filter )
  {
    std::string val;
    if ( ! fact->get_attribute( G_context.opt_filter_attr, val ) )
    {
      // attribute has not been found.
      return;
    }

    if ( ! filter_regex.find( val ) )
    {
      // The attr value does not match the regex.
      return;
    }

  } // end attr filter

  // Print the required fields first
  std::string buf;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf );

  std::string version( "" );
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, version );

  pe_out() << "  Plugin name: " << buf;

  if ( G_context.opt_brief )
  {
    pe_out()  << std::endl;
    return;
  }

  if ( ! version.empty() )
  {
    pe_out() << "      Version: " << version;
  }

  pe_out()  << std::endl;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );
  pe_out() << G_context.m_wtb.wrap_text( buf );

  // Stop here if attributes not enabled
  if ( ! G_context.opt_attrs )
  {
    return;
  }

  buf = "-- Not Set --";
  if ( fact->get_attribute( kwiver::vital::plugin_factory::CONCRETE_TYPE, buf ) )
  {
    buf = kwiver::vital::demangle( buf );
  }
  pe_out() << "      Creates concrete type: " << buf << std::endl;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_FILE_NAME, buf );
  pe_out() << "      Plugin loaded from file: " << buf << std::endl;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, buf );
  pe_out() << "      Plugin module name: " << buf << std::endl;

  // print all the rest of the attributes
  print_functor pf( pe_out() );
  fact->for_each_attr( pf );

  pe_out() << std::endl;
}


// ------------------------------------------------------------------
//
// display full factory
//
void
display_factory( kwiver::vital::plugin_factory_handle_t const fact )
{
  // See if this factory is selected
  if ( G_context.opt_attr_filter )
  {
    std::string val;
    if ( ! fact->get_attribute( G_context.opt_filter_attr, val ) )
    {
      // attribute has not been found.
      return;
    }

    if ( ! filter_regex.find( val ) )
    {
      // The attr value does not match the regex.
      return;
    }

  } // end attr filter

  display_attributes( fact );
}


// ------------------------------------------------------------------
void display_by_category( const kwiver::vital::plugin_map_t& plugin_map,
                          const std::string& category )
{
  kwiver::vital::category_explorer* cat_handler = get_category_handler( category );

  for( auto it : plugin_map )
  {
    std::string ds = kwiver::vital::demangle( it.first );

    kwiver::vital::plugin_factory_vector_t const& facts = it.second;
    if (facts.size() == 0)
    {
      continue;
    }

    kwiver::vital::plugin_factory_handle_t const afact = facts[0];

    // We assume that all factories that support an interface all have the same category.
    std::string cat;
    if ( ! afact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, cat )
         || cat != category )
    {
      continue;
    }

    // If regexp matching is enabled, and this does not match, skip it
    if ( G_context.opt_fact_filt && ( ! fact_regex.find( ds ) ) )
    {
      continue;
    }

    pe_out() << "\nPlugins that implement type \"" << ds << "\"" << std::endl;

    // Get vector of factories
    for( kwiver::vital::plugin_factory_handle_t const fact : facts )
    {
      // If regexp matching is enabled, and this does not match, skip it
      if ( G_context.opt_type_filt )
      {
        std::string type_name = "-- Not Set --";
        if ( ! fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type_name )
             || ( ! type_regex.find( type_name ) ) )
        {
          continue;
        }
      }

      if ( cat_handler )
      {
        cat_handler->explore( fact );
        continue;
      }

      // Default display for factory
      display_factory( fact );

    } // end foreach factory
  } // end interface type

  pe_out() << std::endl;
}


// ------------------------------------------------------------------
kwiver::vital::category_explorer* get_category_handler( const std::string& cat )
{
  std::string handler_name = cat;

  // See if special formatting is requested
  if ( ! G_context.formatting_type.empty() )
  {
    handler_name += "-" + G_context.formatting_type;
  }

  if ( category_map.count( handler_name ) )
  {
    return category_map[handler_name];
  }

  return nullptr;
}


//+ Move this into the context class so it is available to all plugins.
//+ Don't forget to wrap the description text.
// ------------------------------------------------------------------
void print_config( kwiver::vital::config_block_sptr const config )
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();
  const std::string indent( "    " );

  pe_out() << indent << "Configuration block contents\n";

  for( kwiver::vital::config_block_key_t key : all_keys )
  {
    kwiver::vital::config_block_value_t val = config->get_value< kwiver::vital::config_block_value_t > ( key );
    pe_out() << std::endl
             << indent << "\"" << key << "\" = \"" << val << "\"\n";

    kwiver::vital::config_block_description_t descr = config->get_description( key );
    pe_out() << indent << "Description: " << descr << std::endl;
  }
}


// ------------------------------------------------------------------
/**
 * @brief Load plugin explorer plugins
 *
 * Since these plugins are part of the tool, they are loaded separately.
 *
 * @param path Directory of where to look for these plugins.
 */
void load_explorer_plugins( const std::string& path )
{
  LOG_DEBUG( G_logger, "Loading explorer plugins from: " << path );

  // need a dedicated loader to just load the explorer_context files.
  kwiver::vital::plugin_loader pl( "register_explorer_plugin", SHARED_LIB_SUFFIX );

  kwiver::vital::path_list_t pathl;
  const std::string& default_module_paths( DEFAULT_MODULE_PATHS );

  ST::Split( default_module_paths, pathl, PATH_SEPARATOR_CHAR );

  // Check env variable for path specification
  const char * env_ptr = kwiversys::SystemTools::GetEnv( "KWIVER_PLUGIN_PATH" );
  if ( 0 != env_ptr )
  {
    LOG_DEBUG( G_logger, "Adding path(s) \"" << env_ptr << "\" from environment" );
    std::string const extra_module_dirs(env_ptr);

    // Split supplied path into separate items using PATH_SEPARATOR_CHAR as delimiter
    ST::Split( extra_module_dirs, pathl, PATH_SEPARATOR_CHAR );
  }

  // Add our subdirectory to each path element
  for( std::string& p : pathl )
  {
    // This subdirectory must match what is specified in the build system.
    p.append( "/plugin_explorer" );
  }

  // Remove duplicates
  std::sort( pathl.begin(), pathl.end() );
  pathl.erase(std::unique( pathl.begin(), pathl.end()), pathl.end() );

  // Load plugins
  pl.load_plugins( pathl );

  auto fact_list = pl.get_factories( typeid( kwiver::vital::category_explorer ).name() );

  for( auto fact : fact_list )
  {
    std::string name;
    if ( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, name ) )
    {
      auto cat_ex = fact->create_object<kwiver::vital::category_explorer>();
      if ( cat_ex )
      {
        if ( cat_ex->initialize( G_explorer_context ) )
        {
          category_map[name] = cat_ex;
          LOG_DEBUG( G_logger, "Adding category handler for: " << name );
        }
        else
        {
          LOG_DEBUG( G_logger, "Category handler for :" << name << " did not initialize." );
        }
      }
      else
      {
        LOG_WARN( G_logger, "Could not create explorer plugin \"" << name << "\"" );
      }
    }
  }
}


// ------------------------------------------------------------------
int
path_callback( const char*  argument,   // name of argument
               const char*  value,      // value of argument
               void*        call_data ) // data from register call
{
  const std::string p( value );

  G_context.opt_path.push_back( p );
  return 1;   // return true for OK
}


// ==================================================================
/*                   _
 *   _ __ ___   __ _(_)_ __
 *  | '_ ` _ \ / _` | | '_ \
 *  | | | | | | (_| | | | | |
 *  |_| |_| |_|\__,_|_|_| |_|
 *
 */
int
main( int argc, char* argv[] )
{
  // Initialize shared storage
  G_logger = kwiver::vital::get_logger( "plugin_explorer" );
  G_context.m_out_stream = &std::cout; // could use a string stream
  G_context.display_attr = display_attributes; // set display function pointer

  G_explorer_context = new kwiver::vital::context_factory( &G_context );

  {
    const std::string prog_name( argv[0] );

    // Check to see if we are running under a sanctioned alias
    if ( prog_name.find( "processopedia" ) != std::string::npos )
    {
      program_personality = prog_processopedia;
      G_context.opt_process = true;
    }
    else if ( prog_name.find( "algo_explorer" ) != std::string::npos )
    {
      program_personality = prog_alog_explorer;
      G_context.opt_algo = true;
    }
  }

  // Set formatting string for description formatting
  G_context.m_wtb.set_indent_string( "      " );

  // set up the command line args
  G_context.m_args.Initialize( argc, argv );
  G_context.m_args.StoreUnusedArguments( true );
  typedef kwiversys::CommandLineArguments argT;

  G_context.m_args.AddArgument( "--help",    argT::NO_ARGUMENT, &G_context.opt_help, "Display usage information" );
  G_context.m_args.AddArgument( "-h",        argT::NO_ARGUMENT, &G_context.opt_help, "Display usage information" );
  G_context.m_args.AddArgument( "-v",        argT::NO_ARGUMENT, &opt_version, "Display program version" );
  G_context.m_args.AddArgument( "--version", argT::NO_ARGUMENT, &opt_version, "Display program version" );
  G_context.m_args.AddArgument( "--detail",  argT::NO_ARGUMENT, &G_context.opt_detail,
                                "Display detailed information about plugins" );

  G_context.m_args.AddArgument( "-d",        argT::NO_ARGUMENT, &G_context.opt_detail,
                                "Display detailed information about plugins" );

  G_context.m_args.AddArgument( "--path",    argT::NO_ARGUMENT, &G_context.opt_path_list, "Display plugin search path" );
  G_context.m_args.AddCallback( "-I",        argT::SPACE_ARGUMENT, path_callback, 0, "Add directory to plugin search path" );
  G_context.m_args.AddArgument( "--factory", argT::SPACE_ARGUMENT, &G_context.opt_fact_regex,
                                "Only display factories whose interface type matches specified regexp" );

  G_context.m_args.AddArgument( "--fact",    argT::SPACE_ARGUMENT, &G_context.opt_fact_regex,
                                "Only display factories whose interface type matches specified regexp" );

  G_context.m_args.AddArgument( "--type",    argT::SPACE_ARGUMENT, &G_context.opt_type_regex,
                                "Only display factories whose instance name matches the specified regexp. "
                                "The plugins are selected from all available categories." );

  G_context.m_args.AddArgument( "--brief",   argT::NO_ARGUMENT, &G_context.opt_brief, "Generate brief display" );
  G_context.m_args.AddArgument( "-b",        argT::NO_ARGUMENT, &G_context.opt_brief, "Generate brief display" );
  G_context.m_args.AddArgument( "--files",   argT::NO_ARGUMENT, &G_context.opt_files, "Display list of loaded files" );
  G_context.m_args.AddArgument( "--mod",     argT::NO_ARGUMENT, &G_context.opt_modules, "Display list of loaded modules" );
  G_context.m_args.AddArgument( "--all",     argT::NO_ARGUMENT, &G_context.opt_all, "Display all plugin types" );

  std::string algo_arg;
  G_context.m_args.AddArgument( "--algorithm", argT::SPACE_ARGUMENT, &algo_arg,
                                "Display only algorithm type plugins. "
                                "If type is specified as \"all\", then all algorithms are listed. Otherwise, the type "
                                "will be treated as a regexp and only algorithm types that match the regexp will be displayed.");

  G_context.m_args.AddArgument( "--algo",    argT::SPACE_ARGUMENT, &algo_arg,
                                "Display only algorithm type plugins. "
                                "If type is specified as \"all\", then all algorithms are listed. Otherwise, the type "
                                "will be treated as a regexp and only algorithm types that match the regexp will be displayed.");

  std::string proc_arg;
  G_context.m_args.AddArgument( "--process",   argT::SPACE_ARGUMENT, &proc_arg,
                                "Display only sprokit process type plugins. "
                                "If type is specified as \"all\", then all processes are listed. Otherwise, the type "
                                "will be treated as a regexp and only processes names that match the regexp will be displayed." );
  G_context.m_args.AddArgument( "--proc",      argT::SPACE_ARGUMENT, &proc_arg,
                                "Display only sprokit process type plugins. "
                                "If type is specified as \"all\", then all processes are listed. Otherwise, the type "
                                "will be treated as a regexp and only processes names that match the regexp will be displayed." );

  G_context.m_args.AddArgument( "--scheduler", argT::NO_ARGUMENT, &G_context.opt_scheduler, "Display scheduler type plugins" );

  std::vector< std::string > filter_args;
  G_context.m_args.AddArgument( "--filter",  argT::MULTI_ARGUMENT, &filter_args,
                                "Filter factories based on attribute name and value. "
                                "Only two fields must follow: <attr-name> <attr-value>" );

  G_context.m_args.AddArgument( "--summary", argT::NO_ARGUMENT, &G_context.opt_summary,
                                "Display summary of all plugin types" );

  G_context.m_args.AddArgument( "--attrs",   argT::NO_ARGUMENT, &G_context.opt_attrs,
                                "Display raw attributes for plugins without calling any category specific formatting" );

  G_context.m_args.AddArgument( "--load",    argT::SPACE_ARGUMENT, &G_context.opt_load_module,
                                "Load only specified plugin file for inspection. No other plugins are loaded." );

  G_context.m_args.AddArgument( "--skip-relative", argT::NO_ARGUMENT, &G_context.opt_skip_relative,
                                "Skip built-in plugin paths that are relative to the executable location");

  G_context.m_args.AddArgument( "--fmt",     argT::SPACE_ARGUMENT, &G_context.formatting_type,
                                "Generate display using alternative format, such as 'rst' or 'pipe'" );


  // Need to load plugins before we display help since they can add
  // command line options.
  load_explorer_plugins( DEFAULT_MODULE_PATHS );

  // See if there are no args specified. If so, then default to full listing
  if ( argc == 1 )
  {
    G_context.opt_all = true;
  }

  // Parse args
  if ( ! G_context.m_args.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    exit( 0 );
  }

  if ( G_context.opt_help )
  {
    pe_out() << "Usage for " << argv[0] << std::endl
             << "   Version: " << version_string << std::endl
             << G_context.m_args.GetHelp() << std::endl;

    exit( 0 );
  }

  if (opt_version)
  {
    pe_out() << "Version: " << version_string << std::endl;
    // Could also use build date, ...
    exit( 0 );
  }

  // Handle process type parameter
  if ( ! proc_arg.empty() )
  {
    G_context.opt_process = true;

    // Handle process type string
    if ( proc_arg != "all" )
    {
      // if not all, then use as type selector regexp
      G_context.opt_type_filt = true; // do type filtering
      if ( ! type_regex.compile( proc_arg) )
      {
        std::cerr << "Invalid regular expression for type filter \"" << proc_arg << "\"" << std::endl;
        return 1;
      }
    }
  }

  if (! algo_arg.empty() )
  {
    G_context.opt_algo = true;

    // Handle algorithm type string
    if ( algo_arg != "all" )
    {
      // if not all, then use as type selector regexp
      G_context.opt_fact_filt = true; // do factory filtering
      if ( ! fact_regex.compile( algo_arg) )
      {
        std::cerr << "Invalid regular expression for type filter \"" << algo_arg << "\"" << std::endl;
        return 1;
      }
    }

  }

  // If a factory filtering regex specified, then compile it.
  if ( ! G_context.opt_fact_regex.empty() )
  {
    G_context.opt_fact_filt = true;
    if ( ! fact_regex.compile( G_context.opt_fact_regex) )
    {
      std::cerr << "Invalid regular expression for factory filter \"" << G_context.opt_fact_regex << "\"" << std::endl;
      return 1;
    }
  }

  // If a instance type filtering regex specified, then compile it.
  if ( ! G_context.opt_type_regex.empty() )
  {
    G_context.opt_type_filt = true;
    if ( ! type_regex.compile( G_context.opt_type_regex) )
    {
      std::cerr << "Invalid regular expression for type filter \"" << G_context.opt_type_regex << "\"" << std::endl;
      return 1;
    }
  }

  if ( filter_args.size() > 0 )
  {
    // check for attribute based filtering
    if ( filter_args.size() == 2 )
    {
      G_context.opt_attr_filter = true;
      G_context.opt_filter_attr = filter_args[0];
      G_context.opt_filter_regex = filter_args[1];

      if ( ! filter_regex.compile( G_context.opt_filter_regex ) )
      {
        std::cerr << "Invalid regular expression for attribute filter \"" << G_context.opt_filter_regex << "\"" << std::endl;
        return 1;
      }
    }
    else
    {
      std::cerr << "Invalid attribute filtering specification. Two parameters are required." << std::endl;
      return 1;
    }
  }

  // ========
  // Test for incompatible option sets.
  if ( G_context.opt_fact_filt && G_context.opt_attr_filter )
  {
    std::cerr << "Only one of --fact and --filter allowed." << std::endl;
    return 1;
  }

  // test for one of --algorithm or --process allowed
  if ( G_context.opt_algo && G_context.opt_process )
  {
    std::cerr << "Only one of --process or --algorithm allowed" << std::endl;
    return 1;
  }

  //+ test for one of --factory or --type (is this desired?)

  // ========
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  if (!G_context.opt_skip_relative)
  {
    vpm.add_search_path(kwiver::vital::get_executable_path() + "/../lib/modules");
    vpm.add_search_path(kwiver::vital::get_executable_path() + "/../lib/sprokit");
  }

  char** newArgv = 0;
  int newArgc = 0;
  G_context.m_args.GetUnusedArguments(&newArgc, &newArgv);


  // Look for plugin file name from command line
  if ( ! G_context.opt_load_module.empty() )
  {
    // Load file on command line
    local_manager* ll = new(&vpm) local_manager;
    auto loader = ll->loader();

    loader->load_plugin( G_context.opt_load_module );
  }
  else
  {
    // Load from supplied paths and build in paths.
    for( std::string const& path : G_context.opt_path )
    {
      vpm.add_search_path( path );
    }

    vpm.load_all_plugins();
  }

  if ( G_context.opt_path_list )
  {
    pe_out() << "---- Plugin search path\n";

    std::string path_string;
    std::vector< kwiver::vital::path_t > const search_path( vpm.search_path() );
    for( auto module_dir : search_path )
    {
      pe_out() << "    " << module_dir << std::endl;
    }
    pe_out() << std::endl;
  }

  if ( G_context.opt_modules )
  {
    pe_out() << "---- Registered module names:\n";
    auto module_list = vpm.module_map();
    for( auto const name : module_list )
    {
      pe_out() << "   " << name.first << "  loaded from: " << name.second << std::endl;
    }
    pe_out() << std::endl;
  }

  // ------------------------------------------------------------------
  // See if specific category is selected
  if ( G_context.opt_algo )
  {
    auto plugin_map = vpm.plugin_map();
    display_by_category( plugin_map, "algorithm" );
  }

  else if ( G_context.opt_process )
  {
    auto plugin_map = vpm.plugin_map();
    display_by_category( plugin_map, "process" );
  }

  else if ( G_context.opt_scheduler )
  {
    auto plugin_map = vpm.plugin_map();
    display_by_category( plugin_map, "scheduler" );
  }

  // ------------------------------------------------------------------
  // Generate list of factories of any of these options are selected
  else if ( G_context.opt_all
            || G_context.opt_fact_filt
            || G_context.opt_type_filt
            || G_context.opt_detail
            || G_context.opt_brief
            || G_context.opt_attrs
            || G_context.opt_attr_filter )
  {
    auto plugin_map = vpm.plugin_map();

    pe_out() << "\n---- All Registered Plugins\n";

    for( auto it : plugin_map )
    {
      std::string ds = kwiver::vital::demangle( it.first );
      bool first_fact( true );

      // If regexp matching is enabled, and this does not match, skip it
      if ( G_context.opt_fact_filt && ( ! fact_regex.find( ds ) ) )
      {
        continue;
      }

      // Get vector of factories
      kwiver::vital::plugin_factory_vector_t const& facts = it.second;
      for( kwiver::vital::plugin_factory_handle_t const fact : facts )
      {
        // If regexp matching is enabled, and this does not match, skip it
        if ( G_context.opt_type_filt )
        {
          std::string type_name = "-- Not Set --";
          if ( ! fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type_name )
               || ( ! type_regex.find( type_name ) ) )
          {
            continue;
          }
        }

        // See if there is a category handler for this plugin
        std::string category;
        if ( ! G_context.opt_attrs && fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, category ) )
        {
          if ( category_map.count( category ) )
          {
            auto cat_handler = category_map[category];
            cat_handler->explore( fact );
            continue;
          }
        }

        if ( first_fact )
        {
          pe_out() << "\nPlugins that create type \"" << ds << "\"" << std::endl;
          first_fact = false;
        }

        // Default display for factory
        display_factory( fact );

      } // end foreach factory
    } // end interface type

    pe_out() << std::endl;
  }

  //
  // display summary
  //
  if (G_context.opt_summary )
  {
    pe_out() << "\n----Summary of plugin types" << std::endl;
    size_t count(0);

    auto plugin_map = vpm.plugin_map();
    pe_out() << "    " << plugin_map.size() << " types of plugins registered." << std::endl;

    for( auto it : plugin_map )
    {
      std::string ds = kwiver::vital::demangle( it.first );

      // Get vector of factories
      kwiver::vital::plugin_factory_vector_t const& facts = it.second;
      count += facts.size();

      pe_out() << "        " << facts.size() << " plugin(s) that implement \""
               << ds << "\"" <<std::endl;
    } // end interface type

    pe_out() << "    " << count << " total plugins" << std::endl;
  }


  //
  // list files loaded if specified
  //
  if ( G_context.opt_files )
  {
    const auto file_list = vpm.file_list();

    pe_out() << "\n---- Files Successfully Opened" << std::endl;
    for( std::string const& name : file_list )
    {
      pe_out() << "  " << name << std::endl;
    } // end foreach
    pe_out() << std::endl;
  }

  return 0;
} // main
