// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief uv_unwrap_mesh algorithm definition

#ifndef VITAL_ALGO_UV_UNWRAP_MESH_H
#define VITAL_ALGO_UV_UNWRAP_MESH_H

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/mesh_container.h>

namespace kwiver {

namespace vital {

namespace algo {

/// \brief Abstract base class for mesh uv unwrapping.
class VITAL_ALGO_EXPORT uv_unwrap_mesh
  : public kwiver::vital::algorithm
{
public:
  uv_unwrap_mesh();
  PLUGGABLE_INTERFACE( uv_unwrap_mesh );
  /// Unwrap a mesh and generate texture coordinates
  ///
  /// \param mesh [in/out] mesh to unwrap
  virtual void unwrap(
    kwiver::vital::mesh_container_sptr mesh_container ) const = 0;
};

typedef std::shared_ptr< uv_unwrap_mesh > uv_unwrap_mesh_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_UV_UNWRAP_MESH_H
