/*ckwg +29
 * Copyright 2016-2017, 2020 by Kitware, Inc.
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
std::string underline( const std::string& txt, const char c = '=' )
{
  std::string under = std::string( txt.size(), c );
  std::stringstream ss;

  ss << txt << std::endl
     << under << std::endl;
  return ss.str();
}

std::string quoted( const std::string& txt, const char c = '\"' )
{
  std::stringstream ss;
  ss << c << txt << c ;
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

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;

  std::ostream& out_stream() { return m_context->output_stream(); }

  // instance data
  explorer_context* m_context;

  bool opt_hidden;

  kwiver::vital::logger_handle_t m_logger;
}; // end class process_explorer


// ==================================================================
process_explorer::
process_explorer()
  :opt_hidden( false )
  , m_logger( kwiver::vital::get_logger( "process_explorer_plugin" ) )
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

  return true;
}


// ------------------------------------------------------------------
void
process_explorer::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  auto& result = m_context->command_line_result();
  opt_hidden = result["hidden"].as<bool>();

  std::string proc_type = "-- Not Set --";

  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_context->format_description( descrip );

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

  if ( ! pf )
  {
    LOG_ERROR( m_logger, "Could not convert factory to process_factory" );
    return;
  }

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  auto const properties = proc->properties();
  auto const properties_str = join( properties, ", " );

  out_stream()  << "    Properties: " << properties_str << std::endl
                << std::endl;

  // -- config --
  out_stream() << "    -- Configuration --" << std::endl;
  auto const keys = proc->available_config();
  bool config_displayed( false );

  for( auto const & key : keys )
  {
    if ( ! opt_hidden && ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden items
      continue;
    }

    config_displayed = true;
    auto const info = proc->config_info( key );

    auto const& def = info->def;
    auto const conf_desc = m_context->format_description( info->description );
    bool const& tunable = info->tunable;
    char const* const tunable_str = tunable ? "yes" : "no";

    out_stream()  << "    Name       : " << key << std::endl
                  << "    Default    : " << def << std::endl
                  << "    Description: " << conf_desc
                  << "    Tunable    : " << tunable_str << std::endl
                  << std::endl;
  }

  if ( ! config_displayed )
  {
    out_stream() << "    No configuration entries" << std::endl
                 << std::endl;
  }

  // -- input ports --
  out_stream() << "    -- Input ports --" << std::endl;

  auto const iports = proc->input_ports();
  bool iports_empty {true};

  for( auto const & port : iports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden item
      continue;
    }

    auto const info = proc->input_port_info( port );

    auto const& type = info->type;
    auto const& flags = info->flags;
    auto const port_desc = m_context->format_description( info->description );

    auto const flags_str = join( flags, ", " );

    out_stream()  << "    Name       : " << port << std::endl
                  << "    Data type  : " << type << std::endl
                  << "    Flags      : " << flags_str << std::endl
                  << "    Description: " << port_desc << std::endl;

    iports_empty = false;
  }   // end foreach

  if ( iports_empty )
  {
    out_stream() << "    No input ports" << std::endl
                 << std::endl;
  }

  // -- output ports --
  out_stream() << "    -- Output ports --" << std::endl;
  auto const oports = proc->output_ports();

  bool oports_empty {true};

  for( auto const & port : oports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      continue;
    }

    auto const info = proc->output_port_info( port );

    auto const& type = info->type;
    auto const& flags = info->flags;
    auto const port_desc = m_context->format_description( info->description );

    auto const flags_str = join( flags, ", " );

    out_stream()  << "    Name       : " << port << std::endl
                  << "    Data type  : " << type << std::endl
                  << "    Flags      : " << flags_str << std::endl
                  << "    Description: " << port_desc << std::endl;

    oports_empty = false;
  }   // end foreach

    if ( oports_empty )
  {
    out_stream() << "    No output ports" << std::endl
                 <<std::endl;
  }
} // process_explorer::explore

// ==================================================================
/**
 * @brief plugin_explorer support for formatting processes in JSON
 *
 * This class provides the special formatting for processes. It
 * outputs a file that creates a JSON object representing the
 * available processes, their configuration parameters and
 * input and output ports.
 */


/* We possibly want to output JSON format in "VITAL only" mode
 * hence we won't use and JSON libraries for this
 */
#define INDENT_AMOUNT (4)

class json_element
{
public:
  json_element( std::ostream &out_stream, int active_indent, bool first_element=true )
    : m_out_stream( out_stream )
    , m_indent( active_indent + INDENT_AMOUNT )
    , m_first_element(first_element)
    {}
  int indent()
  {
    return m_indent;
  }
  const std::string &comma()
  {
    if ( ! m_first_element )
      return  m_intro_comma;
    else
      return m_no_comma;
  }
  std::ostream &out_stream()
  {
    return m_out_stream;
  }
  ~json_element()
  {
  }
private:
  std::ostream &m_out_stream;
  int m_indent = 0;
  bool m_first_element = true;
  const std::string m_intro_comma = ", ";
  const std::string m_no_comma = "";
};

std::string escape_json(const std::string &s) {
    std::ostringstream o;
    for (auto c = s.cbegin(); c != s.cend(); c++) {
        switch (*c) {
        case '"': o << "\\\""; break;
        case '\\': o << "\\\\"; break;
        case '\b': o << "\\b"; break;
        case '\f': o << "\\f"; break;
        case '\n': o << "\\n"; break;
        case '\r': o << "\\r"; break;
        case '\t': o << "\\t"; break;
        default:
            if ('\x00' <= *c && *c <= '\x1f') {
                o << "\\u"
                  << std::hex << std::setw(4) << std::setfill('0') << (int)*c;
            } else {
                o << *c;
            }
        }
    }
    return o.str();
}

class json_dict
  : public json_element
{
public:
  json_dict( std::ostream &o_stream, int active_indent, bool first_element=true )
    : json_element( o_stream, active_indent, first_element )
  {
    out_stream() << std::string( indent(), ' ' ) << comma() << "{" << std::endl;
  }
  ~json_dict()
  {
    out_stream() << std::string( indent(), ' ') << "}" << std::endl;
  }
};

class json_array
  : public json_element
{
public:
  json_array( std::ostream &o_stream, int active_indent, bool first_element = true )
    : json_element( o_stream, active_indent, first_element )
  {
    out_stream() << std::string( indent(), ' ' ) << comma() << "[" << std::endl;
  }
  ~json_array()
  {
    out_stream() << std::string( indent(), ' ') << "]" << std::endl;
  }
};

template< class ArrayContainerT >
class json_array_items
  : public json_element
{
public:
  json_array_items( std::ostream &o_stream, int active_indent, const ArrayContainerT &arr_items )
    : json_element( o_stream, active_indent, false )
  {
    std::string prefix_comma = "";
    for ( auto item : arr_items )
    {
      out_stream() << std::string( indent(), ' ') << prefix_comma << "\"" << item << "\"" << std::endl;
      prefix_comma = ", ";
    }
  }
};

class json_dict_key
  : public json_element
{
public:
  json_dict_key( std::ostream &o_stream, int active_indent, std::string key, bool first_element = true )
    : json_element( o_stream, active_indent, first_element )
    {
      out_stream() << std::string( indent(), ' ') <<  comma() << quoted(key) << " : " << std::endl;
    }
};

class json_dict_item
  : public json_element
{
public:
  json_dict_item( std::ostream &o_stream, int active_indent, std::string key, std::string value, bool first_element = true )
    : json_element( o_stream, active_indent, first_element )
    {
      out_stream() << std::string( indent(), ' ') <<  comma() << quoted(key) << " : " << quoted(value) << std::endl;
    }
};


class process_explorer_json
  : public category_explorer
{
public:
  process_explorer_json();
  virtual ~process_explorer_json();

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;

  std::ostream& out_stream();

  // instance data
  explorer_context* m_context;
  std::string opt_output_dir;
  std::ofstream m_out_stream;
  std::shared_ptr< json_array > m_root_array;
  bool m_first_process = true;

}; // end class process_explorer_json


// ==================================================================
process_explorer_json::
process_explorer_json()
{
}


process_explorer_json::
~process_explorer_json()
{ }


// ------------------------------------------------------------------
std::ostream&
process_explorer_json::
out_stream()
{
  if ( ! opt_output_dir.empty() )
  {
    return m_out_stream;
  }

  return m_context->output_stream();
}


// ------------------------------------------------------------------
bool
process_explorer_json::
initialize( explorer_context* context )
{
  m_context = context;
  return true;
}


// ------------------------------------------------------------------
void
process_explorer_json::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  if (m_first_process)
  {
    //Opens the first square bracket.
    m_root_array = std::make_shared< json_array >( out_stream(), 0 );
  }
  json_dict plugin_dict( out_stream(), m_root_array->indent(), m_first_process );
  m_first_process = false;
  std::string proc_type = "-- Not Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );
  json_dict_item name_item( out_stream(), plugin_dict.indent(), "proc_type", proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  json_dict_item description_item( out_stream(), plugin_dict.indent(), "description", escape_json(descrip), false );

  std::string proc_class = "-- Not Set --";
  if ( fact->get_attribute( kwiver::vital::plugin_factory::CONCRETE_TYPE, proc_class ) )
  {
    proc_class = kwiver::vital::demangle( proc_class );
  }
  json_dict_item class_item( out_stream(), plugin_dict.indent(), "class_type", proc_class, false );

  // Start the doc page for the process.
  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );
  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  auto const properties = proc->properties();
  if ( properties.size() > 0 )
  {
    json_dict_key property_key_element( out_stream(), plugin_dict.indent(), "properties", false );
    json_array property_array_element ( out_stream(), property_key_element.indent() );
    json_array_items<sprokit::process::properties_t> properties_items( out_stream(), property_array_element.indent(), properties );
  }

  // Configuration Elements
  auto const keys = proc->available_config();
  if ( ! keys.empty() )
  {
    json_dict_key configuration_key_element( out_stream(), plugin_dict.indent(), "configuration", false );
    json_array configuration_array_element ( out_stream(), configuration_key_element.indent() );

    bool first_element = true;
    for( auto const & key : keys )
    {
      if ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden items
        continue;
      }
      auto const info = proc->config_info( key );

      auto def = info->def;
      auto const  conf_desc = info->description;
      bool const& tunable = info->tunable;

      json_dict config_dict( out_stream(), configuration_array_element.indent(), first_element );
      json_dict_item config_key_item( out_stream(), config_dict.indent(), "key", key );
      if (! def.empty() )
      {
        json_dict_item config_def_item( out_stream(), config_dict.indent(), "default", def, false );
      }
      json_dict_item config_tunable_item( out_stream(), config_dict.indent(), "tunable", tunable ? "true" : "false", false );
      json_dict_item config_desc_item( out_stream(), config_dict.indent(), "description", escape_json(conf_desc), false );
      first_element = false;
    }
  }

  // -- input ports --
  auto const iports = proc->input_ports();
  if ( ! iports.empty() )
  {
    json_dict_key iport_key_element( out_stream(), plugin_dict.indent(), "input_ports", false );
    json_array iport_array_element ( out_stream(), iport_key_element.indent() );

    bool first_element = true;
    for( auto const & port : iports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden item
        continue;
      }

      auto const info = proc->input_port_info( port );

      auto const& type = info->type;
      auto const& flags = info->flags;
      auto const port_desc =  info->description;

      json_dict iport_dict( out_stream(), iport_array_element.indent(), first_element );
      json_dict_item iport_name_item( out_stream(), iport_dict.indent(), "name", port );
      json_dict_item iport_type_item( out_stream(), iport_dict.indent(), "type", type, false );
      json_dict_item iport_desc_item( out_stream(), iport_dict.indent(), "description", escape_json(port_desc), false);
      json_dict_key iport_flags_key( out_stream(), iport_dict.indent(), "flags", false );
      json_array iport_flags_array( out_stream(), iport_flags_key.indent() );
      json_array_items<sprokit::process::port_flags_t> iport_flags_array_items( out_stream(), iport_flags_array.indent(), flags );
      first_element = false;
    }   // end foreach
  }

  // -- output ports --
  auto const oports = proc->output_ports();
  if ( ! oports.empty() )
  {
    json_dict_key oport_key_element( out_stream(), plugin_dict.indent(), "output_ports", false );
    json_array oport_array_element ( out_stream(), oport_key_element.indent() );

    bool first_element = true;
    for( auto const & port : oports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden item
        continue;
      }

      auto const info = proc->output_port_info( port );

      auto const& type = info->type;
      auto const& flags = info->flags;
      auto const port_desc =  info->description;

      json_dict oport_dict( out_stream(), oport_array_element.indent(), first_element );
      json_dict_item oport_name_item( out_stream(), oport_dict.indent(), "name", port );
      json_dict_item oport_type_item( out_stream(), oport_dict.indent(), "type", type, false );
      json_dict_item oport_desc_item( out_stream(), oport_dict.indent(), "description", escape_json(port_desc), false);
      json_dict_key oport_flags_key( out_stream(), oport_dict.indent(), "flags", false );
      json_array oport_flags_array( out_stream(), oport_flags_key.indent() );
      json_array_items<sprokit::process::port_flags_t> oport_flags_array_items( out_stream(), oport_flags_array.indent(), flags );
      first_element = false;
    }   // end foreach
  }
} // process_explorer_json::explore


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

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;

  std::ostream& out_stream();
  std::string wrap_rst_text( const std::string& txt );

  // instance data
  explorer_context* m_context;
  kwiver::vital::wrap_text_block m_wtb;
  kwiver::vital::wrap_text_block m_comment_wtb;
  std::string opt_output_dir;
  std::ofstream m_out_stream;

}; // end class process_explorer_rst


// ==================================================================
process_explorer_rst::
process_explorer_rst()
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
  if ( ! opt_output_dir.empty() )
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
  return std::regex_replace( wtxt, std::regex("\n"), std::string(" |br|\\ ") );
}


// ------------------------------------------------------------------
bool
process_explorer_rst::
initialize( explorer_context* context )
{
  m_context = context;

  // Add plugin specific command line option.
  auto& cla = m_context->command_line_args();
  cla.add_options("Process display")
    ("sep-proc-dir", "Output each process as separate files in the specified directory when generating rst format output. "
     "Output is written to stdout if this option is omitted.",
     cxxopts::value<std::string>() )
    ( "hidden", "Display hidden properties and ports." )
  ;

  return true;
}


// ------------------------------------------------------------------
void
process_explorer_rst::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  auto& result = m_context->command_line_result();
  if (result.count("sep-proc-dir"))
  {
    opt_output_dir = result["sep-proc-dir"].as<std::string>();
  }

  std::string proc_type = "-- Not Set --";
  if ( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type )
       && ! opt_output_dir.empty() )
  {
    m_out_stream.open( opt_output_dir + "/" + proc_type + ".rst", std::ios_base::trunc);
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

  // Start the doc page for the process.
  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  auto const properties = proc->properties();
  auto const properties_str = join( properties, ", " );

  // -- config --
  auto const keys = proc->available_config();
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

    for( auto const & key : keys )
    {
      if ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden items
        continue;
      }

      auto const info = proc->config_info( key );

      auto def = info->def;
      auto const  conf_desc =  wrap_rst_text( info->description );
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
  auto const iports = proc->input_ports();
  out_stream() << std::endl << underline( "Input Ports", '-' ) << std::endl;

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

    for( auto const & port : iports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        // skip hidden item
        continue;
      }

      auto const info = proc->input_port_info( port );

      auto const& type = info->type;
      auto const& flags = info->flags;
      auto const port_desc = wrap_rst_text( info->description );

      auto flags_str = join( flags, ", " );
      if ( flags_str.empty() )
      {
        flags_str = "(none)";
      }

      out_stream() << "   \"" << port << "\", \"" << type << "\", \"" <<  flags_str
                   << "\", \"" << port_desc << "\"" << std::endl;
    }   // end foreach
  }

  // -- output ports --
  auto const oports = proc->output_ports();
  out_stream() << std::endl << underline( "Output Ports", '-' ) << std::endl;

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

    for( auto const & port : oports )
    {
      if ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix )
      {
        continue;
      }

      auto const info = proc->output_port_info( port );

      auto const& type = info->type;
      auto const& flags = info->flags;
      auto const port_desc = wrap_rst_text( info->description );

      auto flags_str = join( flags, ", " );
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

  out_stream() << underline( "Pipefile Usage", '-' ) << std::endl
               << "The following sections describe the blocks needed to use this process in a pipe file." << std::endl
               << std::endl
               << underline( "Pipefile block", '-' ) << std::endl
               << ".. code::" << std::endl
               << std::endl

               << " # ================================================================" << std::endl
               << " process <this-proc>" << std::endl
               << "   :: " << proc_type << std::endl;

  // loop over config
  for( auto const & key : keys )
  {
    if ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix )
    {
      // skip hidden items
      continue;
    }

    auto const info = proc->config_info( key );

    auto def = info->def;
    auto const  conf_desc =  m_comment_wtb.wrap_text( info->description );

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
    out_stream() << " # This process will consume the following input ports" << std::endl;

    for( auto const & port : iports )
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
    out_stream() << " # This process will produce the following output ports" << std::endl;

    for( auto const & port : oports )
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
               << "   :members:" << std::endl
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

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;

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

  return true;
}


// ------------------------------------------------------------------
void
process_explorer_pipe::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  auto& result = m_context->command_line_result();
  opt_hidden = result["hidden"].as<bool>();

  std::string proc_type = "-- Not Set --";

  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_wtb.wrap_text( descrip );

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  auto const properties = proc->properties();
  auto const properties_str = join( properties, ", " );

  out_stream() << std::endl
               << "# -----------------------------" << std::endl
               << "process <local-proc-name> :: "  << proc_type << std::endl
               << "#   Properties: " << properties_str << std::endl
               << descrip << std::endl;

  // -- config --
  auto const keys = proc->available_config();

  for( auto const & key : keys )
  {
    if ( ! opt_hidden && ( key.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden items
      continue;
    }

    auto const info = proc->config_info( key );

    auto const& def = info->def;
    auto const& conf_desc =  m_wtb.wrap_text( info->description );
    bool const& tunable = info->tunable;
    char const* const tunable_str = tunable ? "yes" : "no";

    out_stream() << "    " << key << " = " << def << std::endl
                 << conf_desc
                 << "#   Tunable    : " << tunable_str << std::endl
                 << std::endl;
  }

  // -- input ports --
  auto const iports = proc->input_ports();

  for( auto const & port : iports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      // skip hidden item
      continue;
    }

    auto const info = proc->input_port_info( port );

    auto const& type = info->type;
    auto const& flags = info->flags;
    auto const& port_desc = m_wtb.wrap_text( info->description );

    auto const flags_str = join( flags, ", " );

    out_stream() << "    connect from <upstream_port> to " << port << std::endl
                 << "#   Data type  : " << type << std::endl
                 << "#   Flags      : " << flags_str << std::endl
                 << port_desc << std::endl;

  }   // end foreach

  // -- output ports --
  auto const oports = proc->output_ports();

  for( auto const & port : oports )
  {
    if ( ! opt_hidden && ( port.substr( 0, hidden_prefix.size() ) == hidden_prefix ) )
    {
      continue;
    }

    auto const info = proc->output_port_info( port );

    auto const& type = info->type;
    auto const& flags = info->flags;
    auto const& port_desc = m_wtb.wrap_text( info->description );

    auto const flags_str = join( flags, ", " );

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

  fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::process_explorer_json );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "process-json" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Plugin explorer for process category JSON format output" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  vpm.mark_module_as_loaded( module );
}
