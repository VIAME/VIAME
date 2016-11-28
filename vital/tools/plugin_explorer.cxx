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

#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/RegularExpression.hxx>

//+ #include <sprokit/pipeline/process_factory.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

// Global options
bool opt_config( false );
bool opt_detail( false );
bool opt_help( false );
bool opt_path_list( false );
bool opt_brief( false );
bool opt_modules( false );
bool opt_files( false );
bool opt_hidden( false );
bool opt_processes( false );
bool opt_schedulers( false );
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
//
// display full factory list
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
} // display_factory


// ------------------------------------------------------------------
//+ Do we need to document and handle SPROKIT_MODULE_PATH and SPROKIT_CLUSTER_PATH?
//+ this needs to be more like a man page, rather than a short summary
void
print_help()
{
  std::cout << "This program loads vital plugins and displays their data.\n"
            << "Additional paths can be specified in \"KWIVER_PLUGIN_PATH\" environment variable\n"
            << "\n"
            << "Options are:\n"
            << "  --help           displays usage information\n"
            << "  --path           display plugin search path\n"
            << "  -Iname           also load plugins from this directory (can appear multiple times)\n"
            << "  --detail  -d     generate detailed listing\n"
            << "  --fact  regex    display factories that match regexp\n" //+ which attribute is matched?
            << "  --brief          display factory name and description only\n"
            << "  --mod            display list of loaded modules\n"
            << "  --files          display list of files successfully opened to load plugins\n"
  ;

  return;
}


// ------------------------------------------------------------------
void
display_process( kwiver::vital::plugin_factory_handle_t const fact )
{
  // input is proc_type which is really proc name

  std::string proc_type = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );

  if ( opt_brief )
  {
    std::cout << proc_type << ": " << descrip << std::endl;
    return;
  }

  std::cout << "Process type: " << proc_type << std::endl
            << " Description: " << descrip << std::endl;

  // Create the process so we can inspect it.
  sprokit::process_t const proc = sprokit::create_process( proc_type, sprokit::process::name_t() );

  sprokit::process::properties_t const properties = proc->properties();
  std::string const properties_str = join( properties, ", " );

  std::cout << "  Properties: " << properties_str << std::endl
            << "  Configuration:" << std::endl;

  kwiver::vital::config_block_keys_t const keys = proc->available_config();

  // Loop over all config block entries
  VITAL_FOREACH( kwiver::vital::config_block_key_t const & key, keys )
  {
    if ( ! opt_hidden && ( key.find( hidden_prefix ) == 0 ) )
    {
      continue;
    }

    sprokit::process::conf_info_t const info = proc->config_info( key );

    kwiver::vital::config_block_value_t const& def = info->def;
    kwiver::vital::config_block_description_t const& conf_desc = info->description;
    bool const& tunable = info->tunable;
    char const* const tunable_str = tunable ? "yes" : "no";

    std::cout << "    Name       : " << key << std::endl
              << "    Default    : " << def << std::endl
              << "    Description: " << conf_desc << std::endl
              << "    Tunable    : " << tunable_str << std::endl
              << std::endl;
  }

  std::cout << "  - Input ports:" << std::endl;
  sprokit::process::ports_t const iports = proc->input_ports();

  // loop over all input ports
  VITAL_FOREACH( sprokit::process::port_t const & port, iports )
  {
    if ( ! opt_hidden && ( key.find( hidden_prefix ) == 0 ) )
    {
      continue;
    }

    sprokit::process::port_info_t const info = proc->input_port_info( port );

    sprokit::process::port_type_t const& type = info->type;
    sprokit::process::port_flags_t const& flags = info->flags;
    sprokit::process::port_description_t const& port_desc = info->description;

    std::string const flags_str = join( flags, ", " );

    std::cout << "    Name       : " << port << std::endl
              << "    Type       : " << type << std::endl
              << "    Flags      : " << flags_str << std::endl
              << "    Description: " << port_desc << std::endl
              << std::endl;
  }

  std::cout << "  - Output ports:" << std::endl;
  sprokit::process::ports_t const oports = proc->output_ports();

  // Loop over all output ports
  VITAL_FOREACH( sprokit::process::port_t const & port, oports )
  {
    if ( ! opt_hidden && ( key.find( hidden_prefix ) == 0 ) )
    {
      continue;
    }

    sprokit::process::port_info_t const info = proc->output_port_info( port );

    sprokit::process::port_type_t const& type = info->type;
    sprokit::process::port_flags_t const& flags = info->flags;
    sprokit::process::port_description_t const& port_desc = info->description;

    std::string const flags_str = join( flags, ", " );

    std::cout << "    Name       : " << port << std::endl
              << "    Type       : " << type << std::endl
              << "    Flags      : " << flags_str << std::endl
              << "    Description: " << port_desc << std::endl
              << std::endl;
  }

  if ( opt_detail )
  {
    std::cout << std::endl;

    // print all the rest of the attributes
    print_functor pf( std::cout );
    fact->for_each_attr( pf );
  }
            << std::endl;
} // display_process


// ------------------------------------------------------------------
void
display_scheduler( kwiver::vital::plugin_factory_handle_t const fact )
{
  std::string sched_type = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, sched_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );

  std::cout << sched_type << ": " << descrip << std::endl;
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
  arg.AddArgument( "--proc",        argT::NO_ARGUMENT, &opt_processes, "Display list of sprokit processes" );
  arg.AddArgument( "--sched",       argT::NO_ARGUMENT, &opt_schedulers, "Display list of sprokit schedulers" );
  arg.AddArgument( "--hidden",      argT::NO_ARGUMENT, &opt_processes, "Display hidden properties for processes" );
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
    print_help();
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

  // ========
  // Test for incompatible option sets.
  if ( opt_fact_filt && opt_attr_filter )
  {
    std::cerr << "Only one of --fact and --filter allowed." << std::endl;
    return 1;
  }

  // ========
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  VITAL_FOREACH( std::string const& path, opt_path )
  {
    vpm.add_search_path( path );
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

  if ( opt_processes )
  {
    std::cout << "---- Registered processes:\n";
    auto fact_list = vpm.get_factories( typeid( sprokit::process ).name() );
    VITAL_FOREACH( auto a_fact, fact_list )
    {
      display_process( a_fact );
    }
    std::cout << std::endl;
  }

  if (opt_schedulers )
  {
    std::cout << "---- Registered schedulers:\n";
    auto fact_list = vpm.get_factories( typeid( sprokit::scheduler ).name() );
    VITAL_FOREACH( auto a_fact, fact_list )
    {
      display_scheduler( a_fact );
    }
    std::cout << std::endl;
  }

  if ( opt_all )
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
