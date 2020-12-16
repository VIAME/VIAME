// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for mesh uv unwrapping
 */

#ifndef KWIVER_ARROWS_CORE_UV_UNWRAP_MESH_H
#define KWIVER_ARROWS_CORE_UV_UNWRAP_MESH_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/uv_unwrap_mesh.h>
#include <vital/types/mesh.h>
#include <vital/vital_config.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A class for unwrapping a mesh and generating texture coordinates
class KWIVER_ALGO_CORE_EXPORT uv_unwrap_mesh
    : public vital::algo::uv_unwrap_mesh
{
public:
  PLUGIN_INFO( "core",
               "Unwrap a mesh and generate texture coordinates" )

  /// Get configuration
  vital::config_block_sptr get_configuration() const override;

  /// Set configuration
  void set_configuration(vital::config_block_sptr in_config) override;

  /// Check configuration
  bool check_configuration(vital::config_block_sptr config) const override;

  /// Constructor
  uv_unwrap_mesh();

  /// Destructor
  virtual ~uv_unwrap_mesh();

  /// Unwrap a mesh and generate texture coordinate
  /**
   * \param mesh [in/out]
   */
  void unwrap(kwiver::vital::mesh_sptr mesh) const override;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

}
}
}

#endif // KWIVER_ARROWS_CORE_UV_UNWRAP_MESH_H
