/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_vxl_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "format_images_srm_process.h"

// -----------------------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
VIAME_PROCESSES_VXL_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "viame_processes_vxl" );
  kwiver::vital::plugin_factory_handle_t fact_handle;
    if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::vxl::format_images_srm_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::vxl::format_images_srm_process > );

  fact->add_attribute( kvpf::PLUGIN_NAME, "format_images_srm" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Format images in a way optimized for later IQR processing" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );

  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
