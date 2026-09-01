/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_cppdb_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_management/plugin_loader.h>

#include "ingest_descriptors_db_process.h"
#include "fetch_descriptors_db_process.h"
#include "object_track_descriptors_db_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers database-backed processes
 *
 */
extern "C"
VIAME_PROCESSES_CPPDB_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "viame_processes_cppdb" );
    if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::cppdb::ingest_descriptors_db_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::cppdb::ingest_descriptors_db_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "ingest_descriptors_db" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Ingest descriptors into a database" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::cppdb::fetch_descriptors_db_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::cppdb::fetch_descriptors_db_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "fetch_descriptors_db" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Fetch descriptors from database given UIDs" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::cppdb::object_track_descriptors_db_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::cppdb::object_track_descriptors_db_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "object_track_descriptors_db" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Attach descriptors to object track states from database" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
