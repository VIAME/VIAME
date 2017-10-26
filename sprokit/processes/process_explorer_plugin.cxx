/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include <sprokit/processes/process_explorer_plugin_export.h>

#include <vital/tools/explorer_plugin.h>
#include <vital/util/wrap_text_block.h>
#include <vital/util/string.h>

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>

#include <sstream>
#include <iterator>
#include <fstream>
#include <string>
#include <regex>

namespace kwiver {
namespace vital {

namespace {

// ------------------------------------------------------------------
// This is different from the utility string because sometimes we need
// to join a set.
template< class ContainerT >
std::string
join( const ContainerT& vec, const char* delim )
{
  std::stringstream res;
  std::copy( vec.begin(), vec.end(), std::ostream_iterator< std::string > ( res, delim ) );

  // remove trailing delim
  std::string res_str = res.str();
  if (res_str.size() > 1 )
  {
    res_str.erase(res_str.size() - 2 );
  }

  // trim trailing white space
  res_str.erase( res_str.find_last_not_of( " \t\n\r\f\v" ) + 1 );

  return res_str;
}


std::string underline( const std::string& txt, const char c = '=' )
{
  std::string under = std::string( txt.size(), c );
  std::stringstream ss;

  ss << txt << std::endl
     << under << std::endl;
  return ss.str();
}


static std::string const hidden_prefix = "_";

} // end namespace


// ==================================================================
/**
 * @brief plugin_explorer support for formatting processes.
 *
 * This class provides the special formatting for processes.
 */
class process_explorer
  : public category_explorer
{
public:
  process_explorer();
  virtual ~process_explorer();

  virtual bool initialize( explorer_context* context );
  virtual void explore( const kwiver::vital::plugin_factory_handle_t fact );

  std::ostream& out_stream() { return m_context->output_stream(); }

  // instance data
  explorer_context* m_context;

  bool opt_hidden;
}; // end class process_explorer


// ==================================================================
process_explorer::
process_explorer()
  :opt_hidden( false )
{ }


process_explorer::
~process_explorer()
{ }


// ------------------------------------------------------------------
bool
process_explorer::
initialize( explorer_context* context )
{
  m_context = context;

  // Add plugin specific command line option.
  auto cla = m_context->command_line_args();

  // The problem here is that the address of these strings are copied
  // into a control block. This is a problem since they are on the stack. ???
  cla->AddArgument( "--hidden",
                    kwiversys::CommandLineArguments::NO_ARGUMENT,
                    &this->opt_hidden,
                    "Display hidden properties and ports" );
  return true;
}


// ------------------------------------------------------------------
void
process_explorer::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  std::string proc_type = "-- Not Set --";

  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_context->wrap_text( descrip );

  if ( m_context->if_brief() )
  {
    out_stream() << "    Process type: " << proc_type << "   " << descrip << std::endl;
    return;
  }

  out_stream()  << "---------------------\n"
                << "  Process type: " << proc_type << std::endl
                << "  Description: " << descrip << std::endl;

  if ( ! m_context->if_detail() )
  {
    return;
  }

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  sprokit::process::properties_t const properties = proc->properties();
  std::string const properties_str = join( properties, ", " );

  out_stream()  << "    Properties: " << properties_str << std::endl;

  // -- config --
  out_stream() << "    -- Configuration --" << std::endl;
  kwiver::vital::config_block_keys_t const keys = proc->available_config();

  for( kwiver::vital::config_block_key_t const & key : keys )
  {
    if ( ! opt_hidden && ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden items
      continue;
    }

    sprokit::process::conf_info_t const info = proc->config_info( key );

    kwiver::vital::config_block_value_t const& def = info->def;
    kwiver::vital::config_block_description_t const& conf_desc = info->description;
    bool const& tunable = info->tunable;
    char const* const tunable_str = tunable ? "yes" : "no";

    out_stream()  << "    Name       : " << key << std::endl
                  << "    Default    : " << def << std::endl
                  << "    Description: " << conf_desc << std::endl
                  << "    Tunable    : " << tunable_str << std::endl
                  << std::endl;
  }

  // -- input ports --
  out_stream() << "  -- Input ports --" << std::endl;

  sprokit::process::ports_t const iports = proc->input_ports();

  for( sprokit::process::port_t const & port : iports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden item
      continue;
    }

    sprokit::process::port_info_t const info = proc->input_port_info( port );

    sprokit::process::port_type_t const& type = info->type;
    sprokit::process::port_flags_t const& flags = info->flags;
    sprokit::process::port_description_t const& port_desc = info->description;

    std::string const flags_str = join( flags, ", " );

    out_stream()  << "    Name       : " << port << std::endl
                  << "    Data type  : " << type << std::endl
                  << "    Flags      : " << flags_str << std::endl
                  << "    Description: " << port_desc << std::endl
                  << std::endl;
  }   // end foreach

  // -- output ports --
  out_stream() << "  -- Output ports --" << std::endl;
  sprokit::process::ports_t const oports = proc->output_ports();

  for( sprokit::process::port_t const & port : oports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      continue;
    }

    sprokit::process::port_info_t const info = proc->output_port_info( port );

    sprokit::process::port_type_t const& type = info->type;
    sprokit::process::port_flags_t const& flags = info->flags;
    sprokit::process::port_description_t const& port_desc = info->description;

    std::string const flags_str = join( flags, ", " );

    out_stream()  << "    Name       : " << port << std::endl
                  << "    Data type  : " << type << std::endl
                  << "    Flags      : " << flags_str << std::endl
                  << "    Description: " << port_desc << std::endl
                  << std::endl;
  }   // end foreach

  out_stream()  << std::endl;

} // process_explorer::explore


// ==================================================================
/**
 * @brief plugin_explorer support for formatting processes in RST
 *
 * This class provides the special formatting for processes. It creates
 */
class process_explorer_rst
  : public category_explorer
{
public:
  process_explorer_rst();
  virtual ~process_explorer_rst();

  virtual bool initialize( explorer_context* context );
  virtual void explore( const kwiver::vital::plugin_factory_handle_t fact );

  std::ostream& out_stream();
  std::string wrap_rst_text( const std::string& txt );

  // instance data
  explorer_context* m_context;
  kwiver::vital::wrap_text_block m_wtb;
  kwiver::vital::wrap_text_block m_comment_wtb;
  bool opt_files;
  std::ofstream m_out_stream;

}; // end class process_explorer_rst


// ==================================================================
process_explorer_rst::
process_explorer_rst()
  : opt_files( false )
{
  m_wtb.set_indent_string( "" );
  m_comment_wtb.set_indent_string( " # " );
}


process_explorer_rst::
~process_explorer_rst()
{ }


// ------------------------------------------------------------------
std::ostream&
process_explorer_rst::
out_stream()
{
  if (opt_files)
  {
    return m_out_stream;
  }

  return m_context->output_stream();
}


// ------------------------------------------------------------------
std::string
process_explorer_rst::
wrap_rst_text( const std::string& txt )
{
  std::string wtxt = m_wtb.wrap_text( txt );

  // trim trailing whitesapce new line
  wtxt.erase( wtxt.find_last_not_of( " \t\n\r\f\v" ) + 1 );

  return std::regex_replace( wtxt, std::regex("\n"), " |br|\\ " );
}


// ------------------------------------------------------------------
bool
process_explorer_rst::
initialize( explorer_context* context )
{
  m_context = context;

  // Add plugin specific command line option.
  auto cla = m_context->command_line_args();

  static char flag_name[]("--sep-proc-files");
  static char help_text[]("Generate .rst output for processes in separate files." );

  cla->AddArgument( flag_name,
                    kwiversys::CommandLineArguments::NO_ARGUMENT,
                    &this->opt_files,
                    help_text );

  return true;
}


// ------------------------------------------------------------------
void
process_explorer_rst::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  std::string proc_type = "-- Not Set --";
  if ( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type ) && opt_files )
  {
    m_out_stream.open( proc_type + "-config.rst", std::ios_base::trunc);
  }

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = wrap_rst_text( descrip );

  std::string proc_class = "-- Not Set --";
  if ( fact->get_attribute( kwiver::vital::plugin_factory::CONCRETE_TYPE, proc_class ) )
  {
    proc_class = kwiver::vital::demangle( proc_class );
  }

  // output some magic to ehable maintaining newlines
  out_stream() << "  .. |br| raw:: html" << std::endl
               << std::endl
               << "   <br />" << std::endl
               << std::endl;

  out_stream() << underline( proc_type, '=' ) << std::endl
               << descrip << std::endl
               << std::endl;

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  sprokit::process::properties_t const properties = proc->properties();
  std::string const properties_str = join( properties, ", " );

  // -- config --
  kwiver::vital::config_block_keys_t const keys = proc->available_config();
  out_stream() << underline( "Configuration", '-' ) << std::endl;

  if ( keys.empty() )
  {
    out_stream() << "*There are no configuration items for this process.*" << std::endl
                 << std::endl;
  }
  else
  {
    // generate header text
    out_stream() << ".. csv-table::" << std::endl
                 << "   :header: \"Variable\", \"Default\", \"Tunable\", \"Description\"" << std::endl
                 << "   :align: left" << std::endl
                 << "   :widths: auto" << std::endl
                 << std::endl;

    for( kwiver::vital::config_block_key_t const & key : keys )
    {
      if ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden items
        continue;
      }

      sprokit::process::conf_info_t const info = proc->config_info( key );

      kwiver::vital::config_block_value_t def = info->def;
      kwiver::vital::config_block_description_t const  conf_desc =  wrap_rst_text( info->description );
      bool const& tunable = info->tunable;
      char const* const tunable_str = tunable ? "YES" : "NO";

      if ( def.empty() )
      {
        def = "(no default value)";
      }

      out_stream() << "   \"" << key << "\", \"" << def << "\", \"" << tunable_str << "\", \""
                   << conf_desc << "\"" << std::endl;
    }
  }

  // -- input ports --
  sprokit::process::ports_t const iports = proc->input_ports();
  out_stream() << underline( "Input Ports", '-' ) << std::endl;

  if ( iports.empty() )
  {
    out_stream() << "There are no input ports for this process." << std::endl
                 << std::endl;
  }
  else
  {
    out_stream() << ".. csv-table::" << std::endl
                 << "   :header: \"Port name\", \"Data Type\", \"Flags\", \"Description\"" << std::endl
                 << "   :align: left" << std::endl
                 << "   :widths: auto" << std::endl
                 << std::endl;

    for( sprokit::process::port_t const & port : iports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden item
        continue;
      }

      sprokit::process::port_info_t const info = proc->input_port_info( port );

      sprokit::process::port_type_t const& type = info->type;
      sprokit::process::port_flags_t const& flags = info->flags;
      sprokit::process::port_description_t const port_desc = wrap_rst_text( info->description );

      std::string flags_str = join( flags, ", " );
      if ( flags_str.empty() )
      {
        flags_str = "(none)";
      }

      out_stream() << "   \"" << port << "\", \"" << type << "\", \"" <<  flags_str
                   << "\", \"" << port_desc << "\"" << std::endl;
    }   // end foreach
  }

  // -- output ports --
  sprokit::process::ports_t const oports = proc->output_ports();
  out_stream() << underline( "Output Ports", '-' ) << std::endl;

  if ( oports.empty() )
  {
    out_stream() << "There are no output ports for this process." << std::endl
                 << std::endl;
  }
  else
  {
    out_stream() << ".. csv-table::" << std::endl
                 << "   :header: \"Port name\", \"Data Type\", \"Flags\", \"Description\"" << std::endl
                 << "   :align: left" << std::endl
                 << "   :widths: auto" << std::endl
                 << std::endl;

    for( sprokit::process::port_t const & port : oports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        continue;
      }

      sprokit::process::port_info_t const info = proc->output_port_info( port );

      sprokit::process::port_type_t const& type = info->type;
      sprokit::process::port_flags_t const& flags = info->flags;
      sprokit::process::port_description_t const port_desc = wrap_rst_text( info->description );

      std::string flags_str = join( flags, ", " );
      if ( flags_str.empty() )
      {
        flags_str = "(none)";
      }

      out_stream() << "   \"" << port << "\", \"" << type << "\", \"" <<  flags_str
                   << "\", \"" << port_desc << "\"" << std::endl;
    }   // end foreach
  }
  out_stream() << std::endl;

  // ==================================================================
  // -- pipefile usage --

  // Switch to other file
  if ( opt_files )
  {
    m_out_stream.close();
    m_out_stream.open( proc_type + "-pipe.rst", std::ios_base::trunc);
  }

  out_stream() << underline( "Pipefile Usage", '-' ) << std::endl
               << "The following sections describe the blocks needed ot use this process in a pipe file." << std::endl
               << std::endl
               << underline( "Pipefile block", '-' ) << std::endl
               << ".. code::" << std::endl
               << std::endl

               << " # ================================================================" << std::endl
               << " process <this-name>" << std::endl
               << "   :: " << proc_type << std::endl;

  // loop over config
  for( kwiver::vital::config_block_key_t const & key : keys )
  {
    if ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix )
    {
      // skip hidden items
      continue;
    }

    sprokit::process::conf_info_t const info = proc->config_info( key );

    kwiver::vital::config_block_value_t def = info->def;
    kwiver::vital::config_block_description_t const  conf_desc =  m_comment_wtb.wrap_text( info->description );

    if ( def.empty() )
    {
      def = "<value>";
    }

    out_stream() << conf_desc
                 << "   " << key << " = " << def << std::endl;
  } // end for

  out_stream() << " # ================================================================" << std::endl
               << std::endl;

  out_stream() << underline( "Process connections", '~' )
               << std::endl
               << underline( "The following Input ports will need to be set" , '^')
               << ".. code::" << std::endl
               << std::endl;

  // loop over input ports
  if ( iports.empty() )
  {
    out_stream() << " # There are no input port's for this process" << std::endl
                 << std::endl;
  }
  else
  {
    out_stream() << "# This process will consume the following input ports" << std::endl;

    for( sprokit::process::port_t const & port : iports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden item
        continue;
      }

      out_stream() << " connect from <this-proc>." << port << std::endl
                   <<"          to   <upstream-proc>." << port << std::endl;
    }   // end for
  }
  out_stream() << std::endl;


  // loop over output ports
  out_stream() << underline( "The following Output ports will need to be set" , '^')
               << ".. code::" << std::endl
               << std::endl;

  if ( oports.empty() )
  {
    out_stream() << " # There are no output port's for this process" << std::endl
                 << std::endl;
  }
  else
  {
    for( sprokit::process::port_t const & port : oports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        continue;
      }

      out_stream() << " connect from <this-proc>." << port << std::endl
                   <<"          to   <downstream-proc>." << port << std::endl;
    }   // end foreach
  }

  out_stream() << std::endl
               << underline( "Class Description", '-' ) << std::endl
               << ".. doxygenclass:: " << proc_class << std::endl
               << "   :project: kwiver" << std::endl
               << std::endl;

  // close file just in case it was open
  m_out_stream.close();

} // process_explorer_rst::explore


// ==================================================================
/**
 * @brief plugin_explorer support for formatting processes.
 *
 * This class provides the special formatting for processes.
 */
class process_explorer_pipe
  : public category_explorer
{
public:
  process_explorer_pipe();
  virtual ~process_explorer_pipe();

  virtual bool initialize( explorer_context* context );
  virtual void explore( const kwiver::vital::plugin_factory_handle_t fact );

  std::ostream& out_stream() { return m_context->output_stream(); }

  // instance data
  explorer_context* m_context;
  bool opt_hidden;

  // Need special indent prefix so we can not use normal text wrapper.
  kwiver::vital::wrap_text_block m_wtb;

}; // end class process_explorer_pipe


// ==================================================================
process_explorer_pipe::
process_explorer_pipe()
  :opt_hidden( false )
{
  m_wtb.set_indent_string( "#   " );
}


process_explorer_pipe::
~process_explorer_pipe()
{ }


// ------------------------------------------------------------------
bool
process_explorer_pipe::
initialize( explorer_context* context )
{
  m_context = context;

  // Add plugin specific command line option.
  auto cla = m_context->command_line_args();

  // The problem here is that the address of these strings are copied
  // into a control block. This is a problem since they are on the stack. ???
  cla->AddArgument( "--hidden",
                    kwiversys::CommandLineArguments::NO_ARGUMENT,
                    &this->opt_hidden,
                    "Display hidden properties and ports" );
  return true;
}


// ------------------------------------------------------------------
void
process_explorer_pipe::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  std::string proc_type = "-- Not Set --";

  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_wtb.wrap_text( descrip );

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  sprokit::process::properties_t const properties = proc->properties();
  std::string const properties_str = join( properties, ", " );

  out_stream() << std::endl
               << "# -----------------------------" << std::endl
               << "process <local-proc-name> :: "  << proc_type << std::endl
               << "#   Properties: " << properties_str << std::endl
               << descrip << std::endl;

  // -- config --
  kwiver::vital::config_block_keys_t const keys = proc->available_config();

  for( kwiver::vital::config_block_key_t const & key : keys )
  {
    if ( ! opt_hidden && ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden items
      continue;
    }

    sprokit::process::conf_info_t const info = proc->config_info( key );

    kwiver::vital::config_block_value_t const& def = info->def;
    kwiver::vital::config_block_description_t const& conf_desc =  m_wtb.wrap_text( info->description );
    bool const& tunable = info->tunable;
    char const* const tunable_str = tunable ? "yes" : "no";

    out_stream() << "    " << key << " = " << def << std::endl
                 << conf_desc
                 << "#   Tunable    : " << tunable_str << std::endl
                 << std::endl;
  }

  // -- input ports --
  sprokit::process::ports_t const iports = proc->input_ports();

  for( sprokit::process::port_t const & port : iports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden item
      continue;
    }

    sprokit::process::port_info_t const info = proc->input_port_info( port );

    sprokit::process::port_type_t const& type = info->type;
    sprokit::process::port_flags_t const& flags = info->flags;
    sprokit::process::port_description_t const& port_desc = m_wtb.wrap_text( info->description );

    std::string const flags_str = join( flags, ", " );

    out_stream() << "    connect from <upstream_port> to " << port << std::endl
                 << "#   Data type  : " << type << std::endl
                 << "#   Flags      : " << flags_str << std::endl
                 << port_desc << std::endl;

  }   // end foreach

  // -- output ports --
  sprokit::process::ports_t const oports = proc->output_ports();

  for( sprokit::process::port_t const & port : oports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      continue;
    }

    sprokit::process::port_info_t const info = proc->output_port_info( port );

    sprokit::process::port_type_t const& type = info->type;
    sprokit::process::port_flags_t const& flags = info->flags;
    sprokit::process::port_description_t const& port_desc = m_wtb.wrap_text( info->description );

    std::string const flags_str = join( flags, ", " );

    out_stream() << "    connect from " << port << " to <downstream_port>" << std::endl
                 << "#   Data type  : " << type << std::endl
                 << "#   Flags      : " << flags_str << std::endl
                 << port_desc << std::endl;

  }   // end foreach

  out_stream()  << std::endl;

} // process_explorer_pipe::explore


} } // end namespace


// ==================================================================
extern "C"
PROCESS_EXPLORER_PLUGIN_EXPORT
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  static std::string module("process_explorer_plugin" );
  if ( vpm.is_module_loaded( module ) )
  {
    return;
  }

  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::process_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "process" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Plugin explorer for process category." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );


  fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::process_explorer_rst );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "process-rst" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Plugin explorer for process category rst format output" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );


  fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::process_explorer_pipe );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "process-pipe" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Plugin explorer for process category pipeline format output" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

vpm.mark_module_as_loaded( module );
}
