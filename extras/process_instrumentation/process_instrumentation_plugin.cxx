// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "process_instrumentation_plugin_export.h"

#include "timing_process_instrumentation.h"

#include <vital/tools/explorer_plugin.h>
#include <vital/plugin_loader/plugin_loader.h>
#include <vital/util/wrap_text_block.h>
#include <vital/util/string.h>

#include <sstream>
#include <iterator>
#include <fstream>
#include <string>
#include <regex>

namespace kwiver {
namespace vital {

// ==================================================================
/**
 * @brief plugin_explorer support for formatting processes.
 *
 * This class provides the special formatting for processes.
 */
class PROCESS_INSTRUMENTATION_PLUGIN_NO_EXPORT process_instrumentation_explorer
  : public category_explorer
{
public:
  process_instrumentation_explorer();
  virtual ~process_instrumentation_explorer();

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;

  std::ostream& out_stream() { return m_context->output_stream(); }

  // instance data
  explorer_context* m_context;
}; // end class process_instrumentation_explorer

// ==================================================================
process_instrumentation_explorer::
process_instrumentation_explorer()
{ }

process_instrumentation_explorer::
~process_instrumentation_explorer()
{ }

// ------------------------------------------------------------------
bool
process_instrumentation_explorer::
initialize( explorer_context* context )
{
  m_context = context;

  return true;
}

// ------------------------------------------------------------------
void
process_instrumentation_explorer::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  const std::string indent( "    " );
  std::string instr_name = "-- Not Set --";

  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, instr_name );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_context->wrap_text( descrip );

  if ( m_context->if_brief() )
  {
    out_stream() << "    Instrumentation type: " << instr_name << "   " << descrip << std::endl;
    return;
  }

  out_stream()  << "---------------------\n"
                << "  Instrumentation type: " << instr_name << std::endl
                << "  Description         : " << descrip << std::endl;

  if ( ! m_context->if_detail() )
  {
    return;
  }

  auto proc_instr = fact->create_object<sprokit::process_instrumentation>();
  auto config = proc_instr->get_configuration();
  auto all_keys = config->available_values();

  // -- config --
  out_stream() << "    -- Configuration --" << std::endl;
  kwiver::vital::config_block_keys_t const keys = config->available_values();
  bool config_displayed( false );

  for( kwiver::vital::config_block_key_t const & key : all_keys )
  {
    config_displayed = true;
    auto  val = config->get_value< kwiver::vital::config_block_value_t > ( key );

    m_context->output_stream() << indent << "\"" << key << "\" = \"" << val << "\"\n";

    kwiver::vital::config_block_description_t descr = config->get_description( key );
    m_context->output_stream() << indent << "Description: " << m_context->wrap_text( descr ) << std::endl;
  }

  if ( ! config_displayed )
  {
    out_stream() << "    No configuration entries" << std::endl
                 << std::endl;
  }

} // process_instrumentation_explorer::explore

} }

// ==================================================================
extern "C"
PROCESS_INSTRUMENTATION_PLUGIN_EXPORT
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  static std::string module("process_instrumentation_explorer_plugin" );
  if ( vpm.is_module_loaded( module ) )
  {
    return;
  }

  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::process_instrumentation_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "process-instrumentation" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Plugin explorer for process category." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

vpm.mark_module_as_loaded( module );
}
