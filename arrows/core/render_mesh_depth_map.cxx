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
 * \brief Header for core render_mesh_depth_map function
 */

#include "render_mesh_depth_map.h"

#include <vital/types/camera_perspective.h>
#include <vital/types/image.h>
#include <vital/types/vector.h>


namespace {

/// Helper function to check if a point is inside a triangle
bool is_point_inside_triangle(const kwiver::vital::vector_2d& p,
                              const kwiver::vital::vector_2d& a,
                              const kwiver::vital::vector_2d& b,
                              const kwiver::vital::vector_2d& c)
{
  kwiver::vital::vector_2d AB = b - a;
  kwiver::vital::vector_2d AC = c - a;
  kwiver::vital::vector_2d AP = p - a;
  double inv_total_area = 1.0 / (AB[0] * AC[1] - AB[1] * AC[0]);
  double area_1 = inv_total_area * (AB[0] * AP[1] - AB[1] * AP[0]);
  double area_2 = inv_total_area * (AP[0] * AC[1] - AP[1] * AC[0]);
  return area_1 >= 0 && area_2 >= 0 && (area_1 + area_2) <= 1;
}

/// Helper function to compute barycentric coordinates of p w.r.t the triangle abc
kwiver::vital::vector_3d barycentric_coordinates(const kwiver::vital::vector_2d& p,
                                                 const kwiver::vital::vector_2d& a,
                                                 const kwiver::vital::vector_2d& b,
                                                 const kwiver::vital::vector_2d& c)
{
  kwiver::vital::matrix_3x3d abc;
  abc << a, b, c, 1, 1, 1;
  double det_inv = 1.0 / abc.determinant();
  kwiver::vital::vector_3d res;
  res(0) = ((b(1) - c(1)) * (p(0) - c(0)) - (b(0) - c(0)) * (p(1) - c(1))) * det_inv;
  res(1) = ((c(1) - a(1)) * (p(0) - c(0)) - (c(0) - a(0)) * (p(1) - c(1))) * det_inv;
  res(2) = 1.0 - res(0) - res(1);
  return res;
}

}

namespace kwiver {
namespace arrows {


vital::image_container_sptr render_mesh_depth_map(vital::mesh_sptr mesh, vital::camera_perspective_sptr camera)
{
  unsigned int nb_vertices = mesh->num_verts();
  vital::mesh_vertex_array<3>& vertices = dynamic_cast< vital::mesh_vertex_array<3>& >(mesh->vertices());

  // Project all points on image
  std::vector<vital::vector_2d> points_2d(nb_vertices);
  for (unsigned int i = 0; i < vertices.size(); ++i)
  {
    points_2d[i] = camera->project(vertices[i]);
  }

  // Compute the points negative inverse depth
  std::vector<double> points_depth(nb_vertices);
  for (unsigned int i=0; i < vertices.size(); ++i)
  {
    points_depth[i] = -1.0 / camera->depth(vertices[i]);
  }

  // Initialize z_buffer with max infinity
  vital::image_of<double> z_buffer(camera->image_width(), camera->image_height(), 1);
  for (size_t j = 0; j < z_buffer.height(); ++j)
  {
    for (size_t i = 0; i < z_buffer.width(); ++i)
    {
      z_buffer(i, j) = std::numeric_limits<double>::infinity();
    }
  }

  if (mesh->faces().regularity() == 3)
  {
    auto const& triangles = static_cast< const vital::mesh_regular_face_array<3>& >(mesh->faces());
    rasterize_depth(triangles, points_2d, points_depth, z_buffer);
  }
  else
  {
    LOG_ERROR(vital::get_logger("arrows.core.render_mesh_depth_map" ), "The mesh has to be triangular.");
  }

  // Re-compute the real depth
  for (size_t j = 0; j < z_buffer.height(); ++j)
  {
    for (size_t i = 0; i < z_buffer.width(); ++i)
    {
      if (!std::isinf(z_buffer(i, j)))
      {
        z_buffer(i, j) = -1.0 / z_buffer(i, j);
      }
    }
  }
  return std::make_shared<vital::simple_image_container>(z_buffer);
}


void rasterize_depth(const vital::mesh_regular_face_array<3> &triangles,
                     const std::vector<vital::vector_2d> &image_points,
                     const std::vector<double> &depth,
                     vital::image_of<double>& image,
                     bool check_partial_intersection)
{
  int width = image.width();
  int height = image.height();
  // Write faces on z_buffer with depth test
  for (unsigned int f_id = 0; f_id < triangles.size(); ++f_id)
  {
    const vital::vector_2d& a_uv = image_points[triangles(f_id, 0)];
    const vital::vector_2d& b_uv = image_points[triangles(f_id, 1)];
    const vital::vector_2d& c_uv = image_points[triangles(f_id, 2)];

    // skip the face if the three points are outside the image
    if ((a_uv[0] < 0 || a_uv[0] >= width || a_uv[1] < 0 || a_uv[1] >= height) &&
        (b_uv[0] < 0 || b_uv[0] >= width || b_uv[1] < 0 || b_uv[1] >= height) &&
        (c_uv[0] < 0 || c_uv[0] >= width || c_uv[1] < 0 || c_uv[1] >= height))
      continue;
    double a_depth = depth[triangles(f_id, 0)];
    double b_depth = depth[triangles(f_id, 1)];
    double c_depth = depth[triangles(f_id, 2)];

    // rasterization is done over the face axis-aligned bounding box
    int u_min = std::max(0, static_cast<int>(std::round(std::min(a_uv[0], std::min(b_uv[0], c_uv[0])))));
    int u_max = std::min(width - 1, static_cast<int>(std::round(std::max(a_uv[0], std::max(b_uv[0], c_uv[0])))));
    int v_min = std::max(0, static_cast<int>(std::round(std::min(a_uv[1], std::min(b_uv[1], c_uv[1])))));
    int v_max = std::min(height - 1, static_cast<int>(std::round(std::max(a_uv[1], std::max(b_uv[1], c_uv[1])))));

    for (int v = v_min; v <= v_max; ++v)
    {
      for (int u = u_min; u <= u_max; ++u)
      {
        vital::vector_2d p(u, v);

        // Handle pixels on triangle boundaries. Assignment rules:
        //  - if the pixel center is inside the triangle
        //  - if the pixel is not already assigned and if the pixel intersects the triangle
        //    [only if check_partial_intersection]
        bool pixel_belongs_to_triangle = is_point_inside_triangle(p, a_uv, b_uv, c_uv);
        if (check_partial_intersection && !pixel_belongs_to_triangle && std::isinf(image(u, v)))
        {
          // check for pixel - triangle intersection, by sub-sampling points in the pixel
          for (float dy = -0.5; dy <= 0.5; dy += 0.5)
          {
            for (float dx = -0.5; dx <= 0.5; dx += 0.5)
            {
              if (is_point_inside_triangle(p + vital::vector_2d(dx, dy), a_uv, b_uv, c_uv))
              {
                pixel_belongs_to_triangle = true;
                p += vital::vector_2d(dx, dy);
                break;
              }
            }
            if (pixel_belongs_to_triangle) break;
          }
        }

        if (pixel_belongs_to_triangle)
        {
          vital::vector_3d bary_coords = barycentric_coordinates(p, a_uv, b_uv, c_uv);

          double depth = (bary_coords[0] * a_depth +
                          bary_coords[1] * b_depth +
                          bary_coords[2] * c_depth);
          // depth test
          if (depth < image(u, v))
          {
            image(u, v) = depth;
          }
        }
      }
    }
  }
}

}
}
