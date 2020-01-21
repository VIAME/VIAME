/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
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
 * \brief Implementation of mesh operations
 */

#include "mesh_operations.h"
#include <algorithm>
#include <map>
#include <vital/exceptions.h>
#include <vital/logger/logger.h>
#include <Eigen/Geometry>


namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;

/// Subdivide mesh faces into triangle
std::unique_ptr<mesh_regular_face_array<3> >
mesh_triangulate(mesh_face_array_base const& faces)
{
  std::unique_ptr<mesh_regular_face_array<3> >
    tris(new mesh_regular_face_array<3>);
  int group = -1;
  if (faces.has_groups())
  {
    group = 0;
  }
  for (unsigned int f=0; f<faces.size(); ++f)
  {
    for (unsigned i=2; i<faces.num_verts(f); ++i)
    {
      tris->push_back(mesh_tri(faces(f, 0), faces(f, i-1), faces(f, i)));
    }
    if (group >= 0 && f+1 >= faces.groups()[group].second)
    {
      tris->make_group(faces.groups()[group++].first);
    }
  }
  return tris;
}


/// Subdivide quadrilaterals into triangle
std::unique_ptr<mesh_regular_face_array<3> >
mesh_triangulate(mesh_regular_face_array<4> const& faces)
{
  std::unique_ptr<mesh_regular_face_array<3> >
    tris(new mesh_regular_face_array<3>);
  int group = -1;
  if (faces.has_groups())
  {
    group = 0;
  }
  for (unsigned int f=0; f<faces.size(); ++f)
  {
    const mesh_regular_face<4>& face = faces[f];
    tris->push_back(mesh_tri(face[0], face[1], face[2]));
    tris->push_back(mesh_tri(face[0], face[2], face[3]));
    if (group >= 0 && f+1 >= faces.groups()[group].second)
    {
      tris->make_group(faces.groups()[group++].first);
    }
  }
  return tris;
}


/// Triangulate the faces of the mesh (in place)
void
mesh_triangulate(mesh& mesh)
{
  switch (mesh.faces().regularity())
  {
    case 3: break;
    case 4:
    {
      std::unique_ptr<mesh_face_array_base> tris( mesh_triangulate(
        static_cast<mesh_regular_face_array<4> const&>(mesh.faces())));
      mesh.set_faces(std::move(tris));
      break;
    }
    default:
    {
      std::unique_ptr<mesh_face_array_base> tris(mesh_triangulate(mesh.faces()));
      mesh.set_faces(std::move(tris));
      break;
    }
  }
}

namespace {

// Helper function to find intersection of edge between two vertices and a plane.
// If the intersection has already been computed (from the other half edge)
// reuse the previously computed point, otherwise compute a new one.
// Returns the index of the intersection point.
unsigned
edge_plane_intersection(unsigned i1, unsigned i2,
                        vital::mesh_vertex_array<3>& vertices,
                        std::vector<double> const& dist,
                        std::map<std::pair<unsigned, unsigned>, unsigned>& new_vert_map)
{
  vital::vector_3d const& v1 = vertices[i1];
  vital::vector_3d const& v2 = vertices[i2];
  double const& d1 = dist[i1];
  double const& d2 = dist[i2];

  auto key = (i1 > i2) ? std::make_pair(i2, i1)
                       : std::make_pair(i1, i2);

  auto iter = new_vert_map.find(key);
  if (iter != new_vert_map.end())
  {
    return iter->second;
  }

  // epsilon to avoid creating duplicate vertices
  constexpr double eps = 1e-3;
  double lambda = -d1 / (d2 - d1);
  if (lambda <= eps)
  {
    new_vert_map[key] = i1;
    return i1;
  }
  if (lambda >= 1.0 - eps)
  {
    new_vert_map[key] = i2;
    return i2;
  }
  vital::vector_3d isect = v1 + lambda * (v2 - v1);
  unsigned new_vert_ind = vertices.size();
  vertices.push_back(isect);
  new_vert_map[key] = new_vert_ind;

  return new_vert_ind;
}

// choose which diagonal to use when splitting a quad into tris
// pick the combination with the most even area distribution
// if returns true:  use {i1, i2, i4} and {i2, i3, i4}
// if returns false: use {i1, i2, i3} and {i3, i4, i1}
bool
choose_cut(unsigned i1, unsigned i2, unsigned i3, unsigned i4,
           vital::mesh_vertex_array<3>& vertices)
{
  vital::vector_3d const& v1 = vertices[i1];
  vital::vector_3d const& v2 = vertices[i2];
  vital::vector_3d const& v3 = vertices[i3];
  vital::vector_3d const& v4 = vertices[i4];

  double t1 = (v1 - v2).cross(v3 - v2).squaredNorm();
  double t2 = (v2 - v3).cross(v4 - v3).squaredNorm();
  double t3 = (v3 - v4).cross(v1 - v4).squaredNorm();
  double t4 = (v4 - v1).cross(v2 - v1).squaredNorm();

  double r1 = (t1 > t3) ? t1 / t3 : t3 / t1;
  double r2 = (t2 > t4) ? t2 / t4 : t4 / t2;

  return r2 > r1;
}

}


/// Clip a triangular mesh with a plane
bool
clip_mesh(mesh& mesh,
          vector_4d const& plane)
{
  auto& vertices = dynamic_cast< vital::mesh_vertex_array<3>& >(mesh.vertices());

  // check for a triangular mesh
  if (mesh.faces().regularity() != 3)
  {
    LOG_ERROR(vital::get_logger("arrows.core.clip_mesh"),
      "Clipping requires triangular mesh.");
    return false;
  }

  bool clipped = false;
  std::vector<double> dist;
  dist.reserve(vertices.size());
  for (auto const& vert : vertices)
  {
    dist.push_back(vert.homogeneous().dot(plane));
  }

  // keep track of which edge intersection vertices have been added
  std::map<std::pair<unsigned, unsigned>, unsigned> new_vert_map;

  using tri_array = vital::mesh_regular_face_array<3>;
  auto const& triangles = static_cast< const tri_array& >(mesh.faces());
  std::unique_ptr<tri_array> clipped_tris(new tri_array);
  for (auto const& tri : triangles)
  {
    unsigned ind[3] = { tri[0], tri[1], tri[2] };
    // enumerate the 8 possible combinations of which side of the plane
    // the vertices lie on
    uint8_t mode = ((dist[tri[0]] > 0.0) ? 4 : 0)
                 + ((dist[tri[1]] > 0.0) ? 2 : 0)
                 + ((dist[tri[2]] > 0.0) ? 1 : 0);
    // if a triangle crosses the plane (modes 1-6) re-order the indices
    // such that the first (ind[0]) is on the opposite side of the plane
    // from the others (ind[1] and ind[2])
    switch(mode)
    {
    case 7:
      clipped_tris->push_back(tri);
      continue;
    case 0:
      clipped = true;
      continue;
    case 1:
    case 6:
      std::rotate(ind, ind + 2, ind + 3); // [2, 0, 1]
      break;
    case 2:
    case 5:
      std::rotate(ind, ind + 1, ind + 3); // [1, 2, 0]
      break;
    case 3:
    case 4:
      // use the initial order [0, 1, 2]
      break;
    default:
      VITAL_THROW( vital::invalid_data, "invalid triangle clipping case");
    }
    clipped = true;

    if (std::abs(dist[ind[0]]) < 1e-8)
    {
      if (dist[ind[1]] > 0.0)
      {
        clipped_tris->push_back(tri);
      }
      continue;
    }

    // compute (or lookup) new edge-plane intersection vertices
    unsigned new_vert_ind1 = edge_plane_intersection(ind[0], ind[1], vertices,
                                                     dist, new_vert_map);
    unsigned new_vert_ind2 = edge_plane_intersection(ind[0], ind[2], vertices,
                                                     dist, new_vert_map);

    if (new_vert_ind1 == new_vert_ind2)
    {
      continue;
    }

    // add new triangles covering the subset on the positive side of the plane
    if (dist[ind[0]] > 0.0)
    {
      clipped_tris->push_back({ind[0], new_vert_ind1, new_vert_ind2});
    }
    else if (choose_cut(new_vert_ind1, ind[1], ind[2], new_vert_ind2, vertices))
    {
      // add the new edge between ind[1] and new_vert_ind2
      clipped_tris->push_back({ new_vert_ind1, ind[1], new_vert_ind2 });
      clipped_tris->push_back({ ind[1], ind[2], new_vert_ind2 });
    }
    else
    {
      // add the new edge between ind[2] and new_vert_ind1
      clipped_tris->push_back({ new_vert_ind1, ind[1], ind[2]});
      clipped_tris->push_back({ ind[2], new_vert_ind2, new_vert_ind1 });
    }
  }
  mesh.set_faces(std::move(clipped_tris));
  return clipped;
}


/// Clip a triangular mesh with a camera frustum
bool
clip_mesh(kwiver::vital::mesh& mesh,
          kwiver::vital::camera_perspective const& camera,
          double near, double far, double margin)
{
  bool clipped = false;
  vital::matrix_3x4d P = camera.as_matrix();

  // clip side planes
  vital::vector_4d left_plane = P.row(0) + margin * P.row(2);
  vital::vector_4d right_plane = (camera.image_width() + margin) * P.row(2) - P.row(0);
  vital::vector_4d top_plane = P.row(1) + margin * P.row(2);
  vital::vector_4d bottom_plane = (camera.image_height() + margin) * P.row(2) - P.row(1);

  clipped = clip_mesh(mesh, left_plane) || clipped;
  clipped = clip_mesh(mesh, right_plane) || clipped;
  clipped = clip_mesh(mesh, top_plane) || clipped;
  clipped = clip_mesh(mesh, bottom_plane) || clipped;

  // clip near and far planes
  vital::vector_4d principal_plane = P.row(2);
  principal_plane /= principal_plane.head<3>().norm();
  if (std::isfinite(near))
  {
    vital::vector_4d near_plane = principal_plane;
    near_plane[3] -= near;
    clipped = clip_mesh(mesh, near_plane) || clipped;
  }
  if (std::isfinite(far))
  {
    vital::vector_4d far_plane = -principal_plane;
    far_plane[3] += far;
    clipped = clip_mesh(mesh, far_plane) || clipped;
  }

  return clipped;
}

}
}
}
