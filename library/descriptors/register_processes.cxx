/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_descriptors_export.h"
#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "compute_track_descriptors_process.h"
#include "handle_descriptor_request_process.h"
#include "perform_query_process.h"

// -----------------------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
VIAME_PROCESSES_DESCRIPTORS_EXPORT
void
register_factories( kwiver::vital::registry& vpm )
{
  using namespace sprokit;
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "viame_processes_descriptors" );
  kwiver::vital::plugin_factory_handle_t fact_handle;
    if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  // `format_images_srm` registered here until upstream de7d0779f removed it:
  // nothing had used KWA since search indexes stopped writing it. It arrived
  // in this file from `plugins/vxl` when P3 ported it off VXL, which is why
  // the merge of that commit deleted a file upstream no longer had.


  // Imported from sprokit/processes/core in P5-T04, under the names they
  // registered under there.
  //
  // The parameters are spelled unusually because `typeid( x ).name()` is in
  // the body: a parameter called `name` would be substituted inside it.
#define VIAME_REGISTER_PROCESS( process_type, plugin, blurb )             \
  {                                                                      \
    auto* imported = new sprokit::cpp_process_factory(                   \
      typeid( process_type ).name(),                                     \
      sprokit::process::interface_name(),                                \
      sprokit::create_new_process< process_type > );                     \
                                                                         \
    imported->add_attribute( kvpf::PLUGIN_NAME, plugin )                 \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb )                  \
      .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );                     \
                                                                         \
    vpm.add_factory( imported );                                         \
  }

  VIAME_REGISTER_PROCESS(
    kwiver::compute_track_descriptors_process, "compute_track_descriptors",
    "Compute track descriptors on the input tracks or detections." )

  VIAME_REGISTER_PROCESS(
    kwiver::perform_query_process, "perform_query",
    "Perform a query." )

  VIAME_REGISTER_PROCESS(
    kwiver::handle_descriptor_request_process, "handle_descriptor_request",
    "Handle a new descriptor request, producing desired "
    "descriptors on the input." )

#undef VIAME_REGISTER_PROCESS

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
