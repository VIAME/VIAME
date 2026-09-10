/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_descriptors_export.h"
#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "compute_track_descriptors_process.h"
#include "format_images_srm_process.h"
#include "handle_descriptor_request_process.h"
#include "perform_query_process.h"

// -----------------------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
VIAME_PROCESSES_DESCRIPTORS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
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

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::descriptors::format_images_srm_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::descriptors::format_images_srm_process > );

  fact->add_attribute( kvpf::PLUGIN_NAME, "format_images_srm" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Format images in a way optimized for later IQR processing" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );

  vpm.add_factory( fact );


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
