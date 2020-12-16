// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for core render_mesh_depth_map function
 */

#include "render_mesh_depth_map.h"

#include <vital/types/camera_perspective.h>
#include <vital/types/image.h>
#include <vital/types/vector.h>
#include <vital/util/transform_image.h>
#include <memory>

namespace kwiver {
namespace arrows {
namespace core {

/// This function renders a depth map of a triangular mesh seen by a camera
vital::image_container_sptr render_mesh_depth_map(vital::mesh_sptr mesh, vital::camera_perspective_sptr camera)
{
  vital::mesh_vertex_array<3>& vertices = dynamic_cast< vital::mesh_vertex_array<3>& >(mesh->vertices());

  std::vector<vital::vector_2d> points_2d(vertices.size());
  std::vector<double> depths(vertices.size());
  for (unsigned int i = 0; i < vertices.size(); ++i)
  {
    points_2d[i] = camera->project(vertices[i]);
    depths[i] = camera->depth(vertices[i]);
  }

  vital::image_of<double> zbuffer(camera->image_width(), camera->image_height(), 1);
  // fill each pixel with infinity
  transform_image(zbuffer, [](double){ return std::numeric_limits<double>::infinity(); } );

  if (mesh->faces().regularity() == 3)
  {
    auto const& triangles = static_cast< const vital::mesh_regular_face_array<3>& >(mesh->faces());
    for (auto const& tri : triangles)
    {
      double const& d1 = depths[tri[0]];
      double const& d2 = depths[tri[1]];
      double const& d3 = depths[tri[2]];
      // for now, skip any triangle that is even partly behind the camera
      // TODO clip triangles that are partly behind the camera
      if (d1 <= 0.0 || d2 <= 0.0 || d3 <= 0.0)
      {
        continue;
      }

      vital::vector_2d const& v1 = points_2d[tri[0]];
      vital::vector_2d const& v2 = points_2d[tri[1]];
      vital::vector_2d const& v3 = points_2d[tri[2]];

      render_triangle(v1, v2, v3,
                      -1.0 / d1, -1.0 / d2, -1.0 / d3,
                      zbuffer);
    }
    transform_image(zbuffer, [](double d){ return std::isinf(d) ? d : (-1.0 / d); } );
  }
  else
  {
    LOG_ERROR(vital::get_logger("arrows.core.render_mesh_depth_map" ), "The mesh has to be triangular.");
  }
  return std::make_shared<vital::simple_image_container>(zbuffer);
}

/// This function renders a height map of a triangular mesh
vital::image_container_sptr render_mesh_height_map(vital::mesh_sptr mesh, vital::camera_sptr camera)
{
  vital::image_of<double> height_map(camera->image_width(), camera->image_height(), 1);
  // fill each pixel with infinity
  transform_image(height_map, [](double){ return std::numeric_limits<double>::infinity(); } );

  if (mesh->faces().regularity() == 3)
  {
    vital::camera_perspective_sptr perspective_camera = std::dynamic_pointer_cast<vital::camera_perspective>(camera);
    if (perspective_camera)
    {
      vital::image_container_sptr depth_map_container = render_mesh_depth_map(mesh, perspective_camera);
      vital::image_of<double> depth_map(depth_map_container->get_image());
      depth_map_to_height_map(perspective_camera, depth_map, height_map);
      return std::make_shared<vital::simple_image_container>(height_map);
    }
    else
    {
      vital::mesh_vertex_array<3>& vertices = dynamic_cast< vital::mesh_vertex_array<3>& >(mesh->vertices());

      std::vector<vital::vector_2d> points_2d(vertices.size());
      for (unsigned int i = 0; i < vertices.size(); ++i)
      {
        points_2d[i] = camera->project(vertices[i]);
      }

      auto const& triangles = static_cast< const vital::mesh_regular_face_array<3>& >(mesh->faces());
      double h1, h2, h3;
      for (unsigned int f = 0; f < triangles.size(); ++f)
      {
        vital::vector_2d& v1 = points_2d[triangles(f, 0)];
        vital::vector_2d& v2 = points_2d[triangles(f, 1)];
        vital::vector_2d& v3 = points_2d[triangles(f, 2)];

        h1 = -vertices[triangles(f, 0)](2);
        h2 = -vertices[triangles(f, 1)](2);
        h3 = -vertices[triangles(f, 2)](2);

        render_triangle(v1, v2, v3, h1, h2, h3, height_map);
      }
      transform_image(height_map, [](double h){ return std::isinf(h) ? h : -h; } );
    }
  }
  else
  {
    LOG_ERROR(vital::get_logger("arrows.core.render_mesh_depth_map" ), "The mesh has to be triangular.");
  }
  return std::make_shared<vital::simple_image_container>(height_map);
}

/// This function converts a depth map into a height map obtained with a perspective camera
void depth_map_to_height_map(vital::camera_perspective_sptr const& camera,
                             vital::image_of<double>& depth_map,
                             vital::image_of<double>& height_map)
{
  vital::matrix_3x4d const& P = camera->as_matrix();
  vital::vector_3d const& v = P.block<3, 3>(0, 0).inverse().row(2);
  double const o = v.dot(-P.col(3));
  assert(depth_map.depth() == 1);
  height_map.set_size(depth_map.width(), depth_map.height(), 1);
  for (unsigned int j = 0; j < depth_map.height(); ++j)
  {
    for (unsigned int i = 0; i < depth_map.width(); ++i)
    {
      double const& d = depth_map(i, j);
      vital::vector_3d pt(i, j , 1);
      height_map(i, j) = std::isinf(d) ? d : d * v.dot(pt) + o;
    }
  }
}

/// Compute a triangle attribute linear interpolation vector
vital::vector_3d
triangle_attribute_vector(vital::vector_2d const& v1,
                          vital::vector_2d const& v2,
                          vital::vector_2d const& v3,
                          double a1, double a2, double a3)
{
  vital::vector_3d b1(v2.x()-v1.x(), v2.y()-v1.y(), a2 - a1);
  vital::vector_3d b2(v3.x()-v1.x(), v3.y()-v1.y(), a3 - a1);
  vital::vector_3d n = b1.cross(b2);
  return { -n.x()/n.z(),
           -n.y()/n.z(),
           (v1.x() * n.x() + v1.y() * n.y() + a1 * n.z()) / n.z() };
}

/// This functions renders a triangle and fills it with depth
void render_triangle(const vital::vector_2d& v1, const vital::vector_2d& v2, const vital::vector_2d& v3,
                     double depth_v1, double depth_v2, double depth_v3,
                     vital::image_of<double>& depth_img)
{
  triangle_scan_iterator tsi(v1, v2, v3);

  // Linear interpolation depth
  auto Vd = triangle_attribute_vector(v1, v2, v3, depth_v1, depth_v2, depth_v3);

  for (tsi.reset(); tsi.next(); )
  {
    int y = tsi.scan_y();
    if (y < 0 || y >= static_cast<int>(depth_img.height()))
      continue;
    int min_x = std::max(0, tsi.start_x());
    int max_x = std::min(static_cast<int>(depth_img.width()) - 1, tsi.end_x());

    double new_i = Vd.y() * y + Vd.z();
    for (int x = min_x; x <= max_x; ++x)
    {
      double depth = new_i + Vd.x() * x;
      if (depth < depth_img(x, y))
      {
        depth_img(x, y) = depth;
      }
    }
  }
}

}
}
}
