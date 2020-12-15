// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief uv_unwrap_mesh algorithm definition
 */

#ifndef VITAL_ALGO_UV_UNWRAP_MESH_H
#define VITAL_ALGO_UV_UNWRAP_MESH_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>

#include <vital/types/mesh.h>

namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for mesh uv unwrapping.
class VITAL_ALGO_EXPORT uv_unwrap_mesh
    : public kwiver::vital::algorithm_def<uv_unwrap_mesh>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "uv_unwrap_mesh"; }

  /// Unwrap a mesh and generate texture coordinates
  /**
   * \param mesh [in/out] mesh to unwrap
   */
  virtual void unwrap(kwiver::vital::mesh_sptr mesh) const = 0;

protected:
  uv_unwrap_mesh();

};

typedef std::shared_ptr<uv_unwrap_mesh> uv_unwrap_mesh_sptr;

} } }
#endif // VITAL_ALGO_UV_UNWRAP_MESH_H
