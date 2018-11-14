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


namespace kwiver {
namespace arrows {

vital::image_container_sptr render_mesh_depth_map(vital::mesh_sptr mesh, vital::camera_perspective_sptr camera)
{
  vital::mesh_vertex_array<3>& vertices = dynamic_cast< vital::mesh_vertex_array<3>& >(mesh->vertices());

  std::vector<vital::vector_2d> points_2d(vertices.size());
  for (unsigned int i = 0; i < vertices.size(); ++i)
  {
    points_2d[i] = camera->project(vertices[i]);
  }

  vital::image_of<double> zbuffer(camera->image_width(), camera->image_height(), 1);
  vital::image_of<double> img(camera->image_width(), camera->image_height(), 1);
  for (size_t j = 0; j < zbuffer.height(); ++j)
  {
    for (size_t i = 0; i < zbuffer.width(); ++i)
    {
      zbuffer(i, j) = std::numeric_limits<double>::infinity();
      img(i, j) = std::numeric_limits<double>::infinity();
    }
  }

  if (mesh->faces().regularity() == 3)
  {
    auto const& triangles = static_cast< const vital::mesh_regular_face_array<3>& >(mesh->faces());
    double d1, d2, d3;
    for (unsigned int f = 0; f < triangles.size(); ++f)
    {
      vital::vector_2d& v1 = points_2d[triangles(f, 0)];
      vital::vector_2d& v2 = points_2d[triangles(f, 1)];
      vital::vector_2d& v3 = points_2d[triangles(f, 2)];

      d1 = camera->depth(vertices[triangles(f, 0)]);
      d2 = camera->depth(vertices[triangles(f, 1)]);
      d3 = camera->depth(vertices[triangles(f, 2)]);

      render_triangle(v1, v2, v3,
                      d1, d2, d3,
                      d1, d2, d3,
                      zbuffer, img, true);
    }
  }
  else
  {
    LOG_ERROR(vital::get_logger("arrows.core.render_mesh_depth_map" ), "The mesh has to be triangular.");
  }
  return std::make_shared<vital::simple_image_container>(zbuffer);
}


vital::image_container_sptr render_mesh_height_map(vital::mesh_sptr mesh, vital::camera_sptr camera)
{
  vital::mesh_vertex_array<3>& vertices = dynamic_cast< vital::mesh_vertex_array<3>& >(mesh->vertices());

  std::vector<vital::vector_2d> points_2d(vertices.size());
  for (unsigned int i = 0; i < vertices.size(); ++i)
  {
    points_2d[i] = camera->project(vertices[i]);
  }

  vital::image_of<double> zbuffer(camera->image_width(), camera->image_height(), 1);
  vital::image_of<double> img(camera->image_width(), camera->image_height(), 1);
  for (size_t j = 0; j < zbuffer.height(); ++j)
  {
    for (size_t i = 0; i < zbuffer.width(); ++i)
    {
      zbuffer(i, j) = std::numeric_limits<double>::infinity();
      img(i, j) = std::numeric_limits<double>::infinity();
    }
  }

  if (mesh->faces().regularity() == 3)
  {
    auto const& triangles = static_cast< const vital::mesh_regular_face_array<3>& >(mesh->faces());
    double d1, d2, d3;
    double a1, a2, a3;
    vital::camera_perspective* perspective_camera = dynamic_cast<vital::camera_perspective*>(camera.get());
    for (unsigned int f = 0; f < triangles.size(); ++f)
    {
      vital::vector_2d& v1 = points_2d[triangles(f, 0)];
      vital::vector_2d& v2 = points_2d[triangles(f, 1)];
      vital::vector_2d& v3 = points_2d[triangles(f, 2)];

      a1 = vertices[triangles(f, 0)](2);
      a2 = vertices[triangles(f, 1)](2);
      a3 = vertices[triangles(f, 2)](2);
      if (perspective_camera)
      { // for perspective cameras, depth is needed to do a correct interpolation
        d1 = perspective_camera->depth(vertices[triangles(f, 0)]);
        d2 = perspective_camera->depth(vertices[triangles(f, 1)]);
        d3 = perspective_camera->depth(vertices[triangles(f, 2)]);
      }
      else
      { // for rpc cameras, -height is used for depth test
        d1 = -vertices[triangles(f, 0)](2);
        d2 = -vertices[triangles(f, 1)](2);
        d3 = -vertices[triangles(f, 2)](2);
      }

      render_triangle<double>(v1, v2, v3,
                      d1, d2, d3,
                      a1, a2, a3,
                      zbuffer, img, perspective_camera != nullptr);
    }
  }
  else
  {
    LOG_ERROR(vital::get_logger("arrows.core.render_mesh_depth_map" ), "The mesh has to be triangular.");
  }
  return std::make_shared<vital::simple_image_container>(img);
}



}
}
