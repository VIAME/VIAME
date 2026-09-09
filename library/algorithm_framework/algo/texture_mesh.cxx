// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief texture_mesh algorithm instantiation

#include <viame/algorithm_framework/algo/texture_mesh.h>

namespace kwiver {

namespace vital {

namespace algo {

texture_mesh
::texture_mesh()
{
  attach_logger( "algo.texture_mesh" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
