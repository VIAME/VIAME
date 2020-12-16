// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief GDAL algorithm registration implementation
 */

#include <arrows/gdal/kwiver_algo_gdal_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/gdal/image_io.h>

namespace kwiver {
namespace arrows {
namespace gdal {

extern "C"
KWIVER_ALGO_GDAL_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.gdal" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "gdal", kwiver::arrows::gdal::image_io );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Read and write image using GDAL." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver
