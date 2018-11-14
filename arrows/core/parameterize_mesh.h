/*ckwg +29
 * Copyright 2018 by Kitware, SAS.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Header for mesh uv parameterization
 */

#ifndef KWIVER_ARROWS_CORE_PARAMETERIZE_MESH_H
#define KWIVER_ARROWS_CORE_PARAMETERIZE_MESH_H

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_core_export.h>
#include <vital/algo/parameterize_mesh.h>

#include <vital/types/mesh.h>


namespace kwiver {
namespace arrows {
namespace core {

/// Compute a uv parameterization of a mesh
/**
 * @brief This class computes texture coordinates for each face corner of a triangle mesh.
 * The texture coordinates are normalized.
 */

class KWIVER_ALGO_CORE_EXPORT parameterize_mesh
    : public vital::algorithm_impl<parameterize_mesh, vital::algo::parameterize_mesh>
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "core";

  /// Description of the algorithm
  static constexpr char const* description = "Parameterize a mesh with tcoords";

  /// Get configuration
  virtual vital::config_block_sptr get_configuration() const;

  /// Set configuration
  virtual void set_configuration(vital::config_block_sptr in_config);

  /// Check configuration
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Constructor
  parameterize_mesh();

  /// Destructor
  virtual ~parameterize_mesh();

  /// Copy Constructor
  parameterize_mesh(const parameterize_mesh& other);

  /**
   * @brief parameterize a mesh with uv coordinates
   * @param mesh [in/out]
   * @return width and height of the corresponding texture
   */
  virtual std::pair<unsigned int, unsigned int> parameterize(kwiver::vital::mesh_sptr mesh) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

}
}
}

#endif // KWIVER_ARROWS_CORE_MESH_UV_PARAMETERIZATION_H
