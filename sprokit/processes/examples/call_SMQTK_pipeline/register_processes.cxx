// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/pipeline/process_registry.h>

// -- list processes to register --
#include "supply_image.h"
#include "accept_descriptor.h"
#include "smqtk_extract_export.h"

// ----------------------------------------------------------------
/** \brief Regsiter processes
 *
 *
 */
extern "C"
SMQTK_EXTRACT_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "SMQTK_extract" );

  if ( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------
  auto fact = vpm.ADD_PROCESS( kwiver::supply_image );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "supply_image" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "supplies a single image." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( kwiver::accept_descriptor );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "accept_descriptor" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "reads a single vector." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  // - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
