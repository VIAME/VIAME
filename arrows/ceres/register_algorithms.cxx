// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Ceres algorithm registration implementation
 */

#include <arrows/ceres/kwiver_algo_ceres_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/ceres/bundle_adjust.h>
#include <arrows/ceres/optimize_cameras.h>

namespace kwiver {
namespace arrows {
namespace ceres {

extern "C"
KWIVER_ALGO_CERES_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.ceres" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "ceres", kwiver::arrows::ceres::bundle_adjust );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Uses Ceres Solver to bundle adjust camera and landmark parameters.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "ceres", kwiver::arrows::ceres::optimize_cameras );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Uses Ceres Solver to optimize camera parameters" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
