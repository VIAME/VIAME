/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>

#include <sstream>
#include <iterator>

namespace kwiver {
namespace vital {

namespace {

// ------------------------------------------------------------------
template< class ContainerT >
std::string
join( const ContainerT& vec, const char* delim )
{
  std::stringstream res;
  std::copy( vec.begin(), vec.end(), std::ostream_iterator< std::string > ( res, delim ) );

  return res.str();
}

static std::string const hidden_prefix = "_";

} // end namespace


// ----------------------------------------------------------------
/**
 * @brief
 *
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

  if ( ! m_context->if_detail() )
  {
    out_stream() << "  " << proc_type << ": " << descrip << std::endl;
    return;
  }

  out_stream()  << "---------------------\n"
                << "  Process type: " << proc_type << std::endl
                << "  Description: " << descrip << std::endl;

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( fact.get() );

  sprokit::process_t const proc = pf->create_object( kwiver::vital::config_block::empty_config() );

  sprokit::process::properties_t const properties = proc->properties();
  std::string const properties_str = join( properties, ", " );

  out_stream()  << "    Properties: " << properties_str << std::endl
                << "    -- Configuration --" << std::endl;

  kwiver::vital::config_block_keys_t const keys = proc->available_config();

  VITAL_FOREACH( kwiver::vital::config_block_key_t const & key, keys )
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

  out_stream() << "  Input ports:" << std::endl;

  sprokit::process::ports_t const iports = proc->input_ports();

  VITAL_FOREACH( sprokit::process::port_t const & port, iports )
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
                  << "    Type       : " << type << std::endl
                  << "    Flags      : " << flags_str << std::endl
                  << "    Description: " << port_desc << std::endl
                  << std::endl;
  }   // end foreach

  out_stream() << "  Output ports:" << std::endl;

  sprokit::process::ports_t const oports = proc->output_ports();

  VITAL_FOREACH( sprokit::process::port_t const & port, oports )
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
                  << "    Type       : " << type << std::endl
                  << "    Flags      : " << flags_str << std::endl
                  << "    Description: " << port_desc << std::endl
                  << std::endl;
  }   // end foreach

  out_stream()  << std::endl;

} // process_explorer::explore

} } // end namespace


// ==================================================================
extern "C"
PROCESS_EXPLORER_PLUGIN_EXPORT
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::process_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "process" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Plugin explorer for process category." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
}
