/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <plugins/vxl/viame_processes_vxl_export.h>
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "vxl_srm_image_formatter_process.h"

// -----------------------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
VIAME_PROCESSES_VXL_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_vxl" );

  if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  auto fact = vpm.ADD_PROCESS( viame::vxl::vxl_srm_image_formatter_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME,
                        "format_srm_images" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
                    module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Format images in a way optimized for later IQR processing" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
