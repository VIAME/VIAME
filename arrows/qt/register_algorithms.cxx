// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Register Qt algorithms implementation
 */

#include <arrows/qt/kwiver_algo_qt_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/qt/image_io.h>

namespace kwiver {
namespace arrows {
namespace qt {

// ----------------------------------------------------------------------------
extern "C"
KWIVER_ALGO_QT_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  ::kwiver::vital::algorithm_registrar reg( vpm, "arrows.qt" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_algorithm< image_io >();

  reg.mark_module_as_loaded();
}

} // end namespace qt
} // end namespace arrows
} // end namespace kwiver
