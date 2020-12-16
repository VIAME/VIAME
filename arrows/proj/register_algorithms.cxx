// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief PROJ algorithm registration implementation
 */

#include <arrows/proj/geo_conv.h>

#include <arrows/proj/kwiver_algo_proj_plugin_export.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/types/geodesy.h>

namespace kwiver {
namespace arrows {
namespace proj {

extern "C"
KWIVER_ALGO_PROJ_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.proj" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // register geo-conversion functor
  static auto geo_conv = kwiver::arrows::proj::geo_conversion{};
  vital::set_geo_conv( &geo_conv );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace proj
} // end namespace arrows
} // end namespace kwiver
