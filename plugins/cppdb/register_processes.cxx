/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_cppdb_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

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
    if( sprokit::is_process_module_loaded( vpm, "viame_processes_cppdb_export.h" ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::cppdb::ingest_descriptors_db_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::cppdb::ingest_descriptors_db_process > );
  
  // PLUGIN_NAME will be extracted from process or set manually in add_attribute calls below
  // fact->add_attribute( kvpf::PLUGIN_NAME, "name_here" );  
  
  vpm.add_factory( fact );

  fact = vpm.ADD_PROCESS( viame::cppdb::fetch_descriptors_db_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "fetch_descriptors_db" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Fetch descriptors from database given UIDs" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  fact = vpm.ADD_PROCESS( viame::cppdb::object_track_descriptors_db_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "object_track_descriptors_db" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Attach descriptors to object track states from database" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, "viame_processes_cppdb_export.h" );
}
