// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/processes/cluster_explorer_plugin_export.h>

#include <vital/tools/explorer_plugin.h>
#include <vital/util/wrap_text_block.h>
#include <vital/util/string.h>

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline_util/bakery_display.h>
#include <sprokit/pipeline_util/cluster_bakery.h>
#include <sprokit/pipeline_util/cluster_info.h>

#include <sstream>
#include <iterator>
#include <fstream>
#include <string>
#include <regex>

namespace kwiver {
namespace vital {

// ==================================================================
/**
 * @brief plugin_explorer support for formatting clusters.
 *
 * This class provides the special formatting for processes.
 */
class cluster_explorer
  : public category_explorer
{
public:
  cluster_explorer();
  virtual ~cluster_explorer();

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;

  std::ostream& out_stream() { return m_context->output_stream(); }

  // instance data
  explorer_context* m_context;

  bool opt_hidden;

  kwiver::vital::logger_handle_t m_logger;
}; // end class cluster_explorer

// ==================================================================
cluster_explorer::
cluster_explorer()
  :opt_hidden( false )
  , m_logger( kwiver::vital::get_logger( "cluster_explorer_plugin" ) )
{ }

cluster_explorer::
~cluster_explorer()
{ }

// ------------------------------------------------------------------
bool
cluster_explorer::
initialize( explorer_context* context )
{
  m_context = context;

  return true;
}

// ------------------------------------------------------------------
void
cluster_explorer::
explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  auto& result = m_context->command_line_result();
  opt_hidden = result["hidden"].as<bool>();

  std::string proc_type = "-- Not Set --";

  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, proc_type );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_context->format_description( descrip );

  // if detail, call bakery_display(cluster_bakery)

  if ( m_context->if_brief() )
  {
    out_stream() << "    Cluster type: " << proc_type << "   " << descrip << std::endl;
    return;
  }

  if ( ! m_context->if_detail() )
  {
    out_stream()  << "---------------------\n"
                  << "  Cluster type: " << proc_type << std::endl
                  << "  Description: " << descrip << std::endl;
    return;
  }

  // Use bakery display

  // There is a wealth of info associated with a cluster, but it is
  // all buried in the config_info structure.
  sprokit::cluster_process_factory* pf = dynamic_cast< sprokit::cluster_process_factory* > ( fact.get() );
  if ( ! pf ) // cast failed
  {
    return;
  }

  auto cbp = pf->m_cluster_info->m_bakery;;
  sprokit::bakery_display b_disp( out_stream() );
  b_disp.print( *cbp );

  out_stream() << "---- Pipeline Detail ----\n\n";

  sprokit::bakery_base* bb = dynamic_cast< sprokit::bakery_base* >( cbp.get() );
  b_disp.print( *bb );

} // cluster_explorer::explore

}} // end namespace

// ==================================================================
extern "C"
CLUSTER_EXPLORER_PLUGIN_EXPORT
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  static std::string module("cluster_explorer_plugin" );
  if ( vpm.is_module_loaded( module ) )
  {
    return;
  }

  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::cluster_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "cluster" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Plugin explorer for cluster category output" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  vpm.mark_module_as_loaded( module );
}
