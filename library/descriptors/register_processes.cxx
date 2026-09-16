/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_descriptors_export.h"
#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "compute_track_descriptors_process.h"
#include "handle_descriptor_request_process.h"
#include "perform_query_process.h"
#include "create_database_query_process.h"
#include "extract_desc_ids_for_training_process.h"
#include "fetch_descriptors_process.h"
#include "ingest_descriptors_process.h"
#include "object_track_descriptors_process.h"
#include "process_query_process_adaboost.h"
#include "select_database_query_process.h"
#include "write_query_results_as_tracks_process.h"

// -----------------------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
VIAME_PROCESSES_DESCRIPTORS_EXPORT
void
register_factories( viame::registry& vpm )
{
  using namespace viame::pipeline;
  static auto const module_name = viame::plugin_manager::module_t( "viame_processes_descriptors" );
  viame::plugin_factory_handle_t fact_handle;
    if( viame::pipeline::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = viame::plugin_factory;

  // `format_images_srm` registered here until upstream de7d0779f removed it:
  // nothing had used KWA since search indexes stopped writing it. It arrived
  // in this file from the `vxl` plugin when P3 ported it off VXL, which is why
  // the merge of that commit deleted a file upstream no longer had.


  // Imported from sprokit/processes/core in P5-T04, under the names they
  // registered under there.
  //
  // The parameters are spelled unusually because `typeid( x ).name()` is in
  // the body: a parameter called `name` would be substituted inside it.
#define VIAME_REGISTER_PROCESS( process_type, plugin, blurb )             \
  {                                                                      \
    auto* imported = new viame::pipeline::cpp_process_factory(                   \
      typeid( process_type ).name(),                                     \
      viame::pipeline::process::interface_name(),                                \
      viame::pipeline::create_new_process< process_type > );                     \
                                                                         \
    imported->add_attribute( kvpf::PLUGIN_NAME, plugin )                 \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb )                  \
      .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );                     \
                                                                         \
    vpm.add_factory( imported );                                         \
  }

  VIAME_REGISTER_PROCESS(
    viame::compute_track_descriptors_process, "compute_track_descriptors",
    "Compute track descriptors on the input tracks or detections." )

  VIAME_REGISTER_PROCESS(
    viame::perform_query_process, "perform_query",
    "Perform a query." )

  VIAME_REGISTER_PROCESS(
    viame::handle_descriptor_request_process, "handle_descriptor_request",
    "Handle a new descriptor request, producing desired "
    "descriptors on the input." )

  VIAME_REGISTER_PROCESS(
    viame::core::create_database_query_process, "create_database_query",
    "Create a database query from track descriptors" )

  VIAME_REGISTER_PROCESS(
    viame::core::extract_desc_ids_for_training_process, "extract_desc_ids_for_training",
    "Extract descriptor IDs overlapping with groundtruth" )

  VIAME_REGISTER_PROCESS(
    viame::core::fetch_descriptors_process, "fetch_descriptors",
    "Fetch descriptors from file given UIDs" )

  VIAME_REGISTER_PROCESS(
    viame::core::ingest_descriptors_process, "ingest_descriptors",
    "Ingest descriptors from a pipeline and write to file" )

  VIAME_REGISTER_PROCESS(
    viame::core::object_track_descriptors_process, "object_track_descriptors",
    "Attach descriptors to object track states from file" )

  VIAME_REGISTER_PROCESS(
    viame::process_query_process_adaboost, "process_query_adaboost",
    "Process query descriptors using IQR and AdaBoost ranking" )

  VIAME_REGISTER_PROCESS(
    viame::core::select_database_query_process, "select_database_query",
    "Select between two database query inputs" )

  VIAME_REGISTER_PROCESS(
    viame::core::write_query_results_as_tracks_process, "write_query_results_as_tracks",
    "Write query results as object track CSV" )

#undef VIAME_REGISTER_PROCESS

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  viame::pipeline::mark_process_module_as_loaded( vpm, module_name );
}
