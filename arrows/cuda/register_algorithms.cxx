// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Register depth algorithms implementation
 */

#include <arrows/cuda/kwiver_algo_cuda_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/cuda/integrate_depth_maps.h>

namespace kwiver {
namespace arrows {
namespace cuda {

extern "C"
KWIVER_ALGO_CUDA_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "arrows.cuda" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< integrate_depth_maps >();

  reg.mark_module_as_loaded();
}

} // end namespace cuda
} // end namespace arrows
} // end namespace kwiver
