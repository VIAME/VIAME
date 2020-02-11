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

#include <vital/algo/algorithm_factory.h>
#include <vital/applets/cxxopts.hpp>
#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/plugin_loader/plugin_filter_category.h>
#include <vital/plugin_loader/plugin_filter_default.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/demangle.h>
#include <vital/util/get_paths.h>
#include <vital/util/wrap_text_block.h>

#include <kwiversys/RegularExpression.hxx>
#include <kwiversys/SystemTools.hxx>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <memory>
#include <map>

typedef kwiversys::SystemTools ST;

// -- forward definitions --
static void display_attributes( kwiver::vital::plugin_factory_handle_t const fact );
static void display_by_category( const kwiver::vital::plugin_map_t& plugin_map, const std::string& category );
static kwiver::vital::category_explorer_sptr get_category_handler( const std::string& cat );


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

static std::map< const std::string, kwiver::vital::category_explorer_sptr> category_map;

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
  kwiver::vital::category_explorer_sptr cat_handler = get_category_handler( category );

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
kwiver::vital::category_explorer_sptr
get_category_handler( const std::string& cat )
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

  LOG_WARN( G_logger, "Formatting type \"" << G_context.formatting_type
            << "\" for category \"" << cat << "\" not available." );
  return nullptr;
}

// ------------------------------------------------------------------
/**
 * @brief Load plugin explorer plugins
 *
 * Since these plugins are part of the tool, they are loaded separately.
 *
 * @param path Directory of where to look for these plugins.
 */
void load_explorer_plugins()
{
  // need a dedicated loader to just load the explorer_context files.
  kwiver::vital::plugin_loader pl( "register_explorer_plugin", SHARED_LIB_SUFFIX );

  kwiver::vital::path_list_t pathl;
  const std::string& default_module_paths( DEFAULT_MODULE_PATHS );
  LOG_DEBUG( G_logger, "Loading explorer plugins from: " << DEFAULT_MODULE_PATHS );

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

  pl.load_plugins( pathl );

  auto fact_list = pl.get_factories( typeid( kwiver::vital::category_explorer ).name() );

  for( auto fact : fact_list )
  {
    std::string name;
    if ( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, name ) )
    {
      auto cat_ex = kwiver::vital::category_explorer_sptr(fact->create_object<kwiver::vital::category_explorer>());
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

  // Set formatting string for description formatting
  G_context.m_wtb.set_indent_string( "      " );

  G_context.m_cmd_options.reset( new cxxopts::Options( "plugin_explorer",
                                                       "This program display information about plugins." ) );

  G_context.m_cmd_options->add_options()
    ( "h,help", "Display applet usage" )
    ( "v,version", "Display version information" )
    ;

  G_context.m_cmd_options->add_options("Selector")
    ( "p,proc", "Display only sprokit process type plugins. "
      "If type is specified as \"all\", then all processes are listed. Otherwise, the type "
      "will be treated as a regexp and only processes names that match the regexp will be displayed.",
      cxxopts::value<std::string>() )

    ( "i,interface", "Only display plugins whose interface type matches specified regexp",
      cxxopts::value<std::string>() )

    ( "t,type", "Only display factories whose instance name matches the specified regexp. "
              "The plugins are selected from all available interfaces.",
      cxxopts::value<std::string>() )

    ( "a,algo", "Display only algorithm type plugins. "
      "If type is specified as \"all\", then all algorithms are listed. Otherwise, the type "
      "will be treated as a regexp and only algorithm types that match the regexp will be displayed.",
      cxxopts::value<std::string>() )

    ( "scheduler", "Display scheduler type plugins" )

    ( "filter", "Filter based on <attr-name> <attr-value>",
      cxxopts::value<std::vector<std::string>>())

    ( "all", "Display all plugin types" )
    ;

  G_context.m_cmd_options->add_options("Display modifiers")
    ( "d,detail", "Display detailed information about the plugins" )
    ( "b,brief", "Generate a brief display" )
    ( "attrs", "Display raw attributes for plugins without calling any category specific formatting" )
    ( "fmt", "Generate display using alternative format, such as 'rst' or 'pipe'",
      cxxopts::value<std::string>() )
    ;

  G_context.m_cmd_options->add_options("File related")
    ( "skip-relative", "Skip built-in plugin paths that are relative to the executable location")
    ( "I", "Add directory to search path", cxxopts::value<std::vector<std::string>>() )
    ( "load", "Load only specified plugin file for inspection. No other plugins are loaded.",
      cxxopts::value<std::string>() )
    ;

  G_context.m_cmd_options->add_options("Meta")
    ( "path", "Display plugin search path" )
    ( "files", "Display list of loaded files" )
    ( "mod", "Display list of modules loaded" )
    ( "summary", "Display a summary of all loadable modules" )
    ;


  // Need to load plugins before we display help since they can add
  // command line options.
  load_explorer_plugins();

  // See if there are no args specified. If so, then default to full listing
  if ( argc == 1 )
  {
    G_context.opt_all = true;
  }

  // Parse args
  // The parse result has to be created locally due to class design.
  // No default CTOR, copy CTOR or copy operation.
  try
  {
    static cxxopts::ParseResult local_result = G_context.m_cmd_options->parse( argc, argv );
    G_context.m_result = &local_result; // this is the best we can do
  }
  catch ( std::exception & e )
  {
    std::cerr << "Exception while processing command line parameters.\n" << e.what() << std::endl;
    exit(1);
  }

  auto& cmd_args = *G_context.m_result;  // for shorthand access

  G_context.opt_detail = cmd_args["detail"].as<bool>();
  G_context.opt_help = cmd_args["help"].as<bool>();
  G_context.opt_path_list = cmd_args["path"].as<bool>();
  G_context.opt_brief = cmd_args["brief"].as<bool>();
  G_context.opt_modules = cmd_args["mod"].as<bool>();
  G_context.opt_files = cmd_args["files"].as<bool>();
  G_context.opt_all = cmd_args["all"].as<bool>();
  G_context.opt_scheduler = cmd_args["scheduler"].as<bool>();
  G_context.opt_summary = cmd_args["summary"].as<bool>();
  G_context.opt_attrs = cmd_args["attrs"].as<bool>();
  G_context.opt_skip_relative = cmd_args["skip-relative"].as<bool>();

  if ( G_context.opt_help )
  {
    pe_out() << "Usage for " << argv[0] << std::endl
             << "   Version: " << version_string << std::endl
             << G_context.m_cmd_options->help() << std::endl;

    exit( 0 );
  }

  if ( cmd_args.count("fmt"))
  {
    G_context.formatting_type = cmd_args["fmt"].as<std::string>();
  }

  if (opt_version)
  {
    pe_out() << "Version: " << version_string << std::endl;
    // Could also use build date, ...
    exit( 0 );
  }

  // Handle process type parameter
  if ( cmd_args.count("proc") )
  {
    G_context.opt_process = true;
    std::string const proc_arg{ cmd_args["proc"].as<std::string>() };

    // Handle process type string
    if ( proc_arg != "all" )
    {
      // if not all, then use as type selector regexp
      G_context.opt_type_filt = true; // do type filtering
      if ( ! type_regex.compile( proc_arg) )
      {
        std::cerr << "Invalid regular expression for type filter \""
                  << proc_arg << "\"" << std::endl;
        return 1;
      }
    }
  }

  if ( cmd_args.count("algo") )
  {
    G_context.opt_algo = true;
    std::string const algo_arg{ cmd_args["algo"].as<std::string>() };

    // Handle algorithm type string
    if ( algo_arg != "all" )
    {
      // if not all, then use as type selector regexp
      G_context.opt_fact_filt = true; // do factory filtering
      if ( ! fact_regex.compile( algo_arg) )
      {
        std::cerr << "Invalid regular expression for type filter \""
                  << algo_arg << "\"" << std::endl;
        return 1;
      }
    }

  }

  // If a factory filtering regex specified, then compile it.
  if ( cmd_args.count("interface") )
  {
    G_context.opt_fact_filt = true;
    std::string const regex_arg{ cmd_args["interface"].as<std::string>() };

    if ( ! fact_regex.compile( regex_arg) )
    {
      std::cerr << "Invalid regular expression for factory filter \""
                << regex_arg << "\"" << std::endl;
      return 1;
    }
  }

  // If a instance type filtering regex specified, then compile it.
  if ( cmd_args.count("type") )
  {
    G_context.opt_type_filt = true;
    std::string const regex_arg{ cmd_args["type"].as<std::string>() };

    if ( ! type_regex.compile( regex_arg ) )
    {
      std::cerr << "Invalid regular expression for type filter \""
                << regex_arg << "\"" << std::endl;
      return 1;
    }
  }

  if ( cmd_args.count("filter") )
  {
    // check for attribute based filtering
    if ( cmd_args.count("filter") == 2 )
    {
      std::vector< std::string > const filter_args{ cmd_args["filter"].as<std::vector<std::string>>() };

      G_context.opt_attr_filter = true;
      G_context.opt_filter_attr = filter_args[0];
      G_context.opt_filter_regex = filter_args[1];

      if ( ! filter_regex.compile( G_context.opt_filter_regex ) )
      {
        std::cerr << "Invalid regular expression for attribute filter \""
                  << G_context.opt_filter_regex << "\"" << std::endl;
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

  // ========
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  // remove all default plugin filters
  vpm.get_loader()->clear_filters();

  // Add the default filter which checks for duplicate plugins
  kwiver::vital::plugin_filter_handle_t filt = std::make_shared<kwiver::vital::plugin_filter_default>();
  vpm.get_loader()->add_filter( filt );

  if (!G_context.opt_skip_relative)
  {
    // It is somewhat problematic to keep these in sync with the CMake values
    vpm.add_search_path(kwiver::vital::get_executable_path() + "/../lib/kwiver/plugins");
  }

  // Look for plugin file name from command line
  if ( ! G_context.opt_load_module.empty() )
  {
    // Load file on command line
    auto loader = vpm.get_loader();
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
