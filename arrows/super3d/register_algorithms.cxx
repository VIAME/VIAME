// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Register depth algorithms implementation
 */

#include <arrows/super3d/kwiver_algo_super3d_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/super3d/compute_depth.h>

namespace kwiver {
namespace arrows {
namespace super3d {

extern "C"
KWIVER_ALGO_SUPER3D_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.super3d" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM("super3d", kwiver::arrows::super3d::compute_depth);
  fact->add_attribute(kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "compute depth maps from image sequences.")
    .add_attribute(kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name)
    .add_attribute(kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0")
    .add_attribute(kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc.")
    ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
