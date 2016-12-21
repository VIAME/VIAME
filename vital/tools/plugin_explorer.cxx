/*ckwg +29
 * Copyright 2014-2016 by Kitware, Inc.
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

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/util/demangle.h>
#include <vital/vital_foreach.h>
#include <vital/algo/algorithm_factory.h>

#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/RegularExpression.hxx>

//+ #include <sprokit/pipeline/process_factory.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

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
  virtual ~local_manager() { }

  kwiver::vital::plugin_loader* loader() { return get_loader(); }



protected:



private:



}; // end class local_manager



// Global options
bool opt_config( false );
bool opt_detail( false );
bool opt_help( false );
bool opt_path_list( false );
bool opt_brief( false );
bool opt_modules( false );
bool opt_files( false );
bool opt_all( false );

std::vector< std::string > opt_path;

// Fields used for filtering attributes
bool opt_attr_filter( false );
std::string opt_filter_attr;    // attribute name
std::string opt_filter_regex;   // regex for attr value to match.
kwiversys::RegularExpression filter_regex;

// internal option for factory filtering
bool opt_fact_filt( false );
std::string opt_fact_regex;
kwiversys::RegularExpression fact_regex;

static std::string const hidden_prefix = "_";

//+ will need to add support for algorithms unless this is delegated to algo_explorer

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
void display_attributes( kwiver::vital::plugin_factory_handle_t const fact )
{
  // Print the required fields first
  std::string buf;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf );

  std::string version( "" );
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, version );

  std::cout << "  Plugin name: " << buf;
  if ( ! version.empty() )
  {
    std::cout << "      Version: " << version << std::endl;
  }

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );
  std::cout << "      Description: " << buf << std::endl;

  if ( opt_brief )
  {
    return;
  }

  buf = "-- Not Set --";
  if ( fact->get_attribute( kwiver::vital::plugin_factory::CONCRETE_TYPE, buf ) )
  {
    buf = kwiver::vital::demangle( buf );
  }
  std::cout << "      Creates concrete type: " << buf << std::endl;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_FILE_NAME, buf );
  std::cout << "      Plugin loaded from file: " << buf << std::endl;

  buf = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, buf );
  std::cout << "      Plugin module name: " << buf << std::endl;

  if ( opt_detail )
  {
    // print all the rest of the attributes
    print_functor pf( std::cout );
    fact->for_each_attr( pf );
  }
}


// ------------------------------------------------------------------
//
// display full factory
//
void
display_factory( kwiver::vital::plugin_factory_handle_t const fact )
{
  // See if this factory is selected
  if ( opt_attr_filter )
  {
    std::string val;
    if ( ! fact->get_attribute( opt_filter_attr, val ) )
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
void print_config( kwiver::vital::config_block_sptr const config )
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();
  std::string indent( "    " );

  std::cout << indent << "Configuration block contents\n";

  VITAL_FOREACH( kwiver::vital::config_block_key_t key, all_keys )
  {
    kwiver::vital::config_block_value_t val = config->get_value< kwiver::vital::config_block_value_t > ( key );
    std::cout << std::endl
              << indent << "\"" << key << "\" = \"" << val << "\"\n";

    kwiver::vital::config_block_description_t descr = config->get_description( key );
    std::cout << indent << "Description: " << descr << std::endl;
  }
}


// ------------------------------------------------------------------
/**
 * @brief Display detailed information about an algorithm.
 *
 * @param fact Pointer to algorithm factory.
 */
void display_algorithm( kwiver::vital::plugin_factory_handle_t const fact )
{
  kwiver::vital::algorithm_factory* pf = dynamic_cast< kwiver::vital::algorithm_factory* > ( fact.get() );
  if (0 == pf)
  {
    // Wrong type of factory returned.
    std::cout << "Factory for algorithm could not be converted to algorithm_factory type.";
    return;
  }

  kwiver::vital::algorithm_sptr ptr = pf->create_object();
  std::string type = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::INTERFACE_TYPE, type );

  std::string impl = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl );

  std::cout << "\n---------------------\n"
            << "Detailed info on type \"" << type << "\" implementation \"" << impl << "\""
            << std::endl;

  // should this be enabled with the detail option?
  display_attributes( fact );

  if ( opt_config )
  {
    // Get configuration
    auto config = ptr->get_configuration();

    auto all_keys = config->available_values();
    std::string indent( "    " );

    std::cout << indent << "Configuration block contents" << std::endl;

    VITAL_FOREACH( auto  key, all_keys )
    {
      auto  val = config->get_value< kwiver::vital::config_block_value_t > ( key );
      std::cout << std::endl
                << indent << "\"" << key << "\" = \"" << val << "\"\n";

      kwiver::vital::config_block_description_t descr = config->get_description( key );
      std::cout << indent << "Description: " << descr << std::endl;
    }
  }
}


// ------------------------------------------------------------------
//+ Do we need to document and handle SPROKIT_MODULE_PATH and SPROKIT_CLUSTER_PATH?
//+ this needs to be more like a man page, rather than a short summary
void
print_help( const std::string& name )
{
  std::cout << "This program loads vital plugins and displays their data.\n"
            << "Additional paths can be specified in \"KWIVER_PLUGIN_PATH\" environment variable\n"
            << "\n"
            << "Usage: " << name << "[options] [plugin-file}\n"
            << "Options are:\n"
            << "  --help           displays usage information\n"
            << "  --path           display plugin search path\n"
            << "  -Iname           also load plugins from this directory (can appear multiple times)\n"
            << "  --detail  -d     generate detailed listing\n"
            << "  --fact  regex    display factories that match regexp\n" //+ which attribute is matched?
            << "  --brief          display factory name and description only\n"
            << "  --mod            display list of loaded modules\n"
            << "  --files          display list of files successfully opened to load plugins\n"
            << "  --all            display all factories\n"
            << "  --fact attr regex Filter specified attr name value with regex\n"
  ;

  return;
}


// ------------------------------------------------------------------
int
path_callback( const char*  argument,   // name of argument
               const char*  value,      // value of argument
               void*        call_data ) // data from register call
{
  const std::string p( value );

  opt_path.push_back( p );
  return 1;   // return true for OK
}


// ------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  kwiversys::CommandLineArguments arg;

  arg.Initialize( argc, argv );
  arg.StoreUnusedArguments( true );
  typedef kwiversys::CommandLineArguments argT;

  arg.AddArgument( "--help",        argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "-h",            argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "--detail",      argT::NO_ARGUMENT, &opt_detail, "Display detailed information about plugins" );
  arg.AddArgument( "-d",            argT::NO_ARGUMENT, &opt_detail, "Display detailed information about plugins" );
  arg.AddArgument( "--config",      argT::NO_ARGUMENT, &opt_config, "Display configuration information needed by plugins" );
  arg.AddArgument( "--path",        argT::NO_ARGUMENT, &opt_path_list, "Display plugin search path" );
  arg.AddCallback( "-I",            argT::CONCAT_ARGUMENT, path_callback, 0, "Add directory to plugin search path" );
  arg.AddArgument( "--fact",        argT::SPACE_ARGUMENT, &opt_fact_regex, "Filter factories by interface type based on regexp" );
  arg.AddArgument( "--brief",       argT::NO_ARGUMENT, &opt_brief, "Brief display" );
  arg.AddArgument( "--files",       argT::NO_ARGUMENT, &opt_files, "Display list of loaded files" );
  arg.AddArgument( "--mod",         argT::NO_ARGUMENT, &opt_modules, "Display list of loaded modules" );
  arg.AddArgument( "--all",         argT::NO_ARGUMENT, &opt_all, "Display all factories" );

  std::vector< std::string > filter_args;
  arg.AddArgument( "--filter",      argT::MULTI_ARGUMENT, &filter_args, "Filter factories based on attribute value" );


  //+ add options for:
  // schedulers only

  if ( ! arg.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    exit( 0 );
  }

  if ( opt_help )
  {
    print_help( argv[0] );
    exit( 0 );
  }

  // If a factory filtering regex specified, then compile it.
  if ( ! opt_fact_regex.empty() )
  {
    opt_fact_filt = true;
    if ( ! fact_regex.compile( opt_fact_regex) )
    {
      std::cerr << "Invalid regular expression for factory filter \"" << opt_fact_regex << "\"" << std::endl;
      return 1;
    }
  }

  if ( filter_args.size() > 0 )
  {
    // check for attribute based filtering
    if ( filter_args.size() == 2 )
    {
      opt_attr_filter = true;
      opt_filter_attr = filter_args[0];
      opt_filter_regex = filter_args[1];

      if ( ! filter_regex.compile( opt_filter_regex ) )
      {
        std::cerr << "Invalid regular expression for attribute filter \"" << opt_filter_regex << "\"" << std::endl;
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
  if ( opt_fact_filt && opt_attr_filter )
  {
    std::cerr << "Only one of --fact and --filter allowed." << std::endl;
    return 1;
  }

  // ========
  local_manager vpm;

  char** newArgv = 0;
  int newArgc = 0;
  arg.GetUnusedArguments(&newArgc, &newArgv);

  // Look for plugin file name from command line
  if ( newArgc > 1 )
  {
    // Load file on command line
    auto loader = vpm.loader();

    for ( int i = 1; i < newArgc; ++i )
    {
      loader->load_plugin( newArgv[i] );
    }
  }
  else
  {
    // Load from supplied paths and build in paths.
    VITAL_FOREACH( std::string const& path, opt_path )
    {
      vpm.add_search_path( path );
    }

    vpm.load_all_plugins();
  }

  if ( opt_path_list )
  {
    std::cout << "---- Plugin search path\n";

    std::string path_string;
    std::vector< kwiver::vital::path_t > const search_path( vpm.search_path() );
    VITAL_FOREACH( auto module_dir, search_path )
    {
      std::cout << "    " << module_dir << std::endl;
    }
    std::cout << std::endl;
  }

  if ( opt_modules )
  {
    std::cout << "---- Registered module names:\n";
    auto module_list = vpm.module_map();
    VITAL_FOREACH( auto const name, module_list )
    {
      std::cout << "   " << name.first << "  loaded from: " << name.second << std::endl;
    }
    std::cout << std::endl;
  }

  // Generate list of factories of any of these options are selected
  if ( opt_all || opt_fact_filt || opt_detail || opt_brief || opt_attr_filter )
  {
    auto plugin_map = vpm.plugin_map();

    std::cout << "\n---- All Registered Factories\n";

    VITAL_FOREACH( auto it, plugin_map )
    {
      std::string ds = kwiver::vital::demangle( it.first );

      // If regexp matching is enabled, and this does not match, skip it
      if ( opt_fact_filt && ( ! fact_regex.find( ds ) ) )
      {
        continue;
      }

      std::cout << "\nFactories that create type \"" << ds << "\"" << std::endl;

      // Get vector of factories
      kwiver::vital::plugin_factory_vector_t const& facts = it.second;
      VITAL_FOREACH( kwiver::vital::plugin_factory_handle_t const fact, facts )
      {
        display_factory( fact );
      } // end factory
    } // end interface type
    std::cout << std::endl;
  }

  //
  // list files is specified
  //
  if ( opt_files )
  {
    const auto file_list = vpm.file_list();

    std::cout << "\n---- Files Successfully Opened" << std::endl;
    VITAL_FOREACH( std::string const& name, file_list )
    {
      std::cout << "  " << name << std::endl;
    } // end foreach
    std::cout << std::endl;
  }

  return 0;
} // main
