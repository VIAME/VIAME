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
 * \brief Implementation of the mesh uv parameterization
 */

#include "parameterize_mesh.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <Eigen/Dense>
#include <vital/exceptions.h>

using namespace kwiver::vital;

namespace {

/// This structure is used to represent a 2D triangle
struct triangle_t
{
  kwiver::vital::vector_2d a;
  kwiver::vital::vector_2d b;
  kwiver::vital::vector_2d c;
  unsigned int face_id;
  unsigned int height;
  unsigned int width;
};
}

namespace kwiver {
namespace arrows {
namespace core {

/// Private implementation class
class parameterize_mesh::priv
{
public:
  /// Constructor
  priv()
  {
  }

  priv(const priv& other)
  {
  }

  double resolution;
  unsigned int interior_margin;
  unsigned int exterior_margin;
};

// Constructor
parameterize_mesh::parameterize_mesh()
  : d_(new priv)
{
  attach_logger( "arrows.core.parameterize_mesh" );
}

// Copy constructor
parameterize_mesh
::parameterize_mesh(const parameterize_mesh& other)
  : d_(new priv(*other.d_))
{
}


// Destructor
parameterize_mesh
::~parameterize_mesh()
{
}


// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
parameterize_mesh::get_configuration() const
{
  vital::config_block_sptr config =
      vital::algo::parameterize_mesh::get_configuration();
  config->set_value("resolution", d_->resolution,
                    "Resolution used to link texture units (pixel) to mesh units (meter, ...)"
                    "It is expressed in mesh units / pixel.");

  config->set_value("interior_margin", d_->interior_margin,
                    "It is the margin between triangles on each axis (in pixels).");

  config->set_value("exterior_margin", d_->exterior_margin,
                    "It is the exterior margin placed on each border of the texture.");
  return config;
}


// Set the configuration
void parameterize_mesh
::set_configuration(vital::config_block_sptr in_config)
{
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config(in_config);

  d_->resolution = config->get_value<double>("resolution", d_->resolution);
  d_->interior_margin = config->get_value<unsigned int>("interior_margin", d_->interior_margin);
  d_->exterior_margin = config->get_value<unsigned int>("exterior_margin", d_->exterior_margin);
}


// Check that the algorithm's configuration vital::config_block is valid
bool parameterize_mesh
::check_configuration(vital::config_block_sptr config) const
{
  return config->has_value("resolution") && config->has_value("interior_margin")
      && config->has_value("exterior_margin");
}


/// Parameterize a mesh
std::pair<unsigned int, unsigned int>
parameterize_mesh::parameterize(kwiver::vital::mesh_sptr mesh) const
{
  if (mesh->faces().regularity() != 3)
  {
    VITAL_THROW( algorithm_exception, this->type_name(), this->impl_name(),
                 "This algorithm expects a regular mesh with triangular faces.");
  }

  const mesh_face_array_base& faces = mesh->faces();
  const mesh_vertex_array<3>& vertices = dynamic_cast<const mesh_vertex_array<3>&>(mesh->vertices());

  // Map each triangle in 2d. The longest edge should be horizontal and its left point is (0, 0)
  std::vector<triangle_t> triangles(mesh->num_faces());
  double total_area = 0.0;
  for (unsigned int f = 0; f < mesh->num_faces(); ++f)
  {
    // face 3d points
    vector_3d pt1 = vertices[faces(f, 0)];
    vector_3d pt2 = vertices[faces(f, 1)];
    vector_3d pt3 = vertices[faces(f, 2)];

    // triangle edges
    vector_3d pt1pt2 = pt2 - pt1;
    vector_3d pt1pt3 = pt3 - pt1;
    vector_3d pt2pt3 = pt3 - pt2;

    if (pt1pt2.norm() < 1e-5 || pt1pt3.norm() < 1e-5 || pt2pt3.norm() < 1e-5 || (pt1pt2.cross(pt1pt3)).norm() < 1e-5)
    {
      triangles[f] = {{0, 0}, {0, 0}, {0, 0}, f, 0, 0};
    }
    else
    {
      // find the longest edge and assign it to AB, C is the other point
      vector_3d AB, AC;
      int longest_edge;
      if (pt1pt2.norm() >= pt1pt3.norm() && pt1pt2.norm() >= pt2pt3.norm())
      {
        // pt1 is A, pt2 is B, pt3 is C
        AB = pt1pt2;
        AC = pt1pt3;
        longest_edge = 0;
      }
      else if (pt2pt3.norm() >= pt1pt3.norm())
      {
        // pt1 is C, pt2 is A, pt3 is B
        AB = pt2pt3;
        AC = -pt1pt2;
        longest_edge = 1;
      }
      else
      {
        // pt1 is B, pt2 is C, pt3 is A
        AB = -pt1pt3;
        AC = -pt2pt3;
        longest_edge = 2;
      }
      // transform the face to 2d
      vector_2d a(0.0, 0.0);
      vector_2d b(AB.norm(), 0.0);
      double proj = AC.dot(AB) / AB.norm();
      vector_2d c(proj, (AC - proj * AB.normalized()).norm());
      // scale B and C by the resolution
      b /= d_->resolution;
      c /= d_->resolution;

      total_area += 0.5 * (b(0) * c(1));
      unsigned int w = static_cast<unsigned int>(std::floor(std::max(b(0), c(0)))) + 1;
      unsigned int h = static_cast<unsigned int>(std::floor(c(1))) + 1;
      if (longest_edge == 0)
      {
        // pt1 is A, pt2 is B, pt3 is C
        triangles[f] = {a, b, c, f, h, w};
      }
      else if (longest_edge == 1)
      {
        // pt1 is C, pt2 is A, pt3 is B
        triangles[f] = {c, a, b, f, h, w};
      }
      else if (longest_edge == 2)
      {
        // pt1 is B, pt2 is C, pt3 is A
        triangles[f] = {b, c, a, f, h, w};
      }
    }
  }

  // Sort triangles by height
  std::vector<unsigned int> face_indices(triangles.size(), 0);
  std::iota(face_indices.begin(), face_indices.end(), 0);

  std::sort(face_indices.begin(), face_indices.end(), [&triangles](unsigned int i, unsigned int j)
  {
    return triangles[i].height < triangles[j].height;
  });

  // Pack triangles
  std::vector<vector_2d> tcoords(mesh->num_faces() * 3);

  // approximate max_width to have a more or less square texture atlas
  const unsigned int max_width = static_cast<unsigned int>(std::ceil(sqrt(total_area * 2)));

  unsigned int current_u = d_->exterior_margin;
  unsigned int current_v = d_->exterior_margin;
  unsigned int next_v = current_v;
  vital::vector_2d shift(0.0, 0.0);
  unsigned int max_u = 0, max_v = 0;
  for (unsigned int f : face_indices)
  {
    if (current_u + triangles[f].width + d_->exterior_margin >= max_width)
    {
      current_u = d_->exterior_margin;
      current_v = next_v + d_->interior_margin;
    }
    shift(0) = current_u;
    shift(1) = current_v;
    triangles[f].a += shift;
    triangles[f].b += shift;
    triangles[f].c += shift;

    if (current_u + triangles[f].width > max_u)
      max_u = current_u + triangles[f].width;
    if (current_v + triangles[f].height > max_v)
      max_v = current_v + triangles[f].height;

    next_v = std::max(next_v, current_v + triangles[f].height);
    current_u += triangles[f].width + d_->interior_margin;
  }

  unsigned int width = max_u + d_->exterior_margin;
  unsigned int height = max_v + d_->exterior_margin;

  // Normalize texture coordinates
  vector_2d norm(width, height);
  for (unsigned int i = 0; i < triangles.size(); ++i)
  {
    tcoords[i * 3 + 0] = triangles[i].a.cwiseQuotient(norm);
    tcoords[i * 3 + 1] = triangles[i].b.cwiseQuotient(norm);
    tcoords[i * 3 + 2] = triangles[i].c.cwiseQuotient(norm);
  }
  mesh->set_tex_coords(tcoords);

  return std::make_pair(width, height);
}

}
}
}
