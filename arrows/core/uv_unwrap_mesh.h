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
    : public vital::algorithm_impl<uv_unwrap_mesh, vital::algo::uv_unwrap_mesh>
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "core";

  /// Description of the algorithm
  static constexpr char const* description = "Unwrap a mesh and generate texture coordinates";

  /// Get configuration
  virtual vital::config_block_sptr get_configuration() const;

  /// Set configuration
  virtual void set_configuration(vital::config_block_sptr in_config);

  /// Check configuration
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Constructor
  uv_unwrap_mesh();

  /// Destructor
  virtual ~uv_unwrap_mesh();

  /// Copy Constructor
  uv_unwrap_mesh(const uv_unwrap_mesh& other);

  /// Unwrap a mesh and generate texture coordinate
  /**
   * \param mesh [in/out]
   */
  virtual void unwrap(kwiver::vital::mesh_sptr mesh) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

}
}
}

#endif // KWIVER_ARROWS_CORE_UV_UNWRAP_MESH_H
