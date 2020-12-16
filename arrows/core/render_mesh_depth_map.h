// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of kwiver::arrows::render_mesh_depth_map function
 */

#ifndef KWIVER_ARROWS_CORE_RENDER_MESH_DEPTH_MAP_H
#define KWIVER_ARROWS_CORE_RENDER_MESH_DEPTH_MAP_H

#include <arrows/core/kwiver_algo_core_export.h>
#include <arrows/core/triangle_scan_iterator.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/image_container.h>
#include <vital/types/mesh.h>

namespace kwiver {
namespace arrows {
namespace core {

/// This function renders a depth map of a triangular mesh seen by a camera
/**
 * \param mesh [in]
 * \param camera [in]
 * \return a depth map
 */
KWIVER_ALGO_CORE_EXPORT
vital::image_container_sptr render_mesh_depth_map(kwiver::vital::mesh_sptr mesh,
                                                  kwiver::vital::camera_perspective_sptr camera);

/// This function renders a height map of a triangular mesh
/**
 * \param mesh [in]
 * \param camera [in]
 * \return height map
 */
KWIVER_ALGO_CORE_EXPORT
vital::image_container_sptr render_mesh_height_map(kwiver::vital::mesh_sptr mesh,
                                                   kwiver::vital::camera_sptr camera);

/// This function converts a depth map into a height map obtained with a perspective camera
/**
 * \param camera [in]
 * \param depth_map [in]
 * \param height_map [out]
 */
KWIVER_ALGO_CORE_EXPORT
void depth_map_to_height_map(vital::camera_perspective_sptr const& camera,
                             vital::image_of<double>& depth_map,
                             vital::image_of<double>& height_map);

/// This functions renders a triangle and fills it with depth
/**
 * \param v1 [in] 2D triangle point
 * \param v2 [in] 2D triangle point
 * \param v3 [in] 2D triangle point
 * \param depth_v1 [in] corresponding depth
 * \param depth_v2 [in] corresponding depth
 * \param depth_v3 [in] corresponding depth
 * \param depth_img [in/out] depth map used and updated
 */
KWIVER_ALGO_CORE_EXPORT
void render_triangle(const vital::vector_2d& v1, const vital::vector_2d& v2, const vital::vector_2d& v3,
                     double depth_v1, double depth_v2, double depth_v3,
                     vital::image_of<double>& depth_img);

/// Compute a triangle attribute linear interpolation vector
/**
  * \param v1 [in]  2D triangle vertex
  * \param v2 [in]  2D triangle vertex
  * \param v3 [in]  2D triangle vertex
  * \param a1 [in]  attribute value associated with v1
  * \param a2 [in]  attribute value associated with v2
  * \param a3 [in]  attribute value associated with v3
  * \returns a 3D vector V such that the dot product of V and (x,y,1) is the
  *          interpolated attribute value at location (x,y)
  */
vital::vector_3d
triangle_attribute_vector(vital::vector_2d const& v1,
                          vital::vector_2d const& v2,
                          vital::vector_2d const& v3,
                          double a1, double a2, double a3);

/// This function renders a triangle and linearly interpolating attributes
/**
 * \param v1 [in] 2D triangle point
 * \param v2 [in] 2D triangle point
 * \param v3 [in] 2D triangle point
 * \param depth_v1 [in] corresponding depth
 * \param depth_v2 [in] corresponding depth
 * \param depth_v3 [in] corresponding depth
 * \param attrib_v1 [in] attribute which is interpolated
 * \param attrib_v2 [in] attribute which is interpolated
 * \param attrib_v3 [in] attribute which is interpolated
 * \param depth_img [in/out] depth map used and updated during depth test
 * \param img [out] image on which the triangle is rendered
 */
template<class T>
void render_triangle(const vital::vector_2d& v1, const vital::vector_2d& v2, const vital::vector_2d& v3,
                     double depth_v1, double depth_v2, double depth_v3,
                     T attrib_v1, T attrib_v2, T attrib_v3,
                     vital::image_of<double>& depth_img,
                     vital::image_of<T>& img)
{
  triangle_scan_iterator tsi(v1, v2, v3);
  double attrib_v1_d = static_cast<double>(attrib_v1);
  double attrib_v2_d = static_cast<double>(attrib_v2);
  double attrib_v3_d = static_cast<double>(attrib_v3);

  // Linear interpolation attributes
  auto Va = triangle_attribute_vector(v1, v2, v3, attrib_v1_d, attrib_v2_d, attrib_v3_d);
  // Linear interpolation depth
  auto Vd = triangle_attribute_vector(v1, v2, v3, depth_v1, depth_v2, depth_v3);

  for (tsi.reset(); tsi.next(); )
  {
    int y = tsi.scan_y();
    if (y < 0 || y >= static_cast<int>(img.height()))
      continue;
    int min_x = std::max(0, tsi.start_x());
    int max_x = std::min(static_cast<int>(img.width()) - 1, tsi.end_x());

    double new_i = Va.y() * y + Va.z();
    double new_i_d = Vd.y() * y + Vd.z();
    for (int x = min_x; x <= max_x; ++x)
    {
      double attrib = new_i + Va.x() * x;
      double depth = new_i_d + Vd.x() * x;
      if (depth < depth_img(x, y))
      {
        img(x, y) =  static_cast<T>(attrib);
        depth_img(x, y) = depth;
      }
    }
  }
}

/// This functions renders a triangle and fills every pixel with value where depth_img is updated
/**
 * \param v1 [in] 2D triangle point
 * \param v2 [in] 2D triangle point
 * \param v3 [in] 2D triangle point
 * \param depth_v1 [in] corresponding depth
 * \param depth_v2 [in] corresponding depth
 * \param depth_v3 [in] corresponding depth
 * \param value [in] value used to fill the triangle
 * \param depth_img [in/out] depth map used and updated during depth test
 */
template<class T>
void render_triangle(const vital::vector_2d& v1, const vital::vector_2d& v2, const vital::vector_2d& v3,
                     double depth_v1, double depth_v2, double depth_v3,
                     T const& value,
                     vital::image_of<double>& depth_img,
                     vital::image_of<T>& img)
{
  triangle_scan_iterator tsi(v1, v2, v3);

  // Linear interpolation depth
  auto Vd = triangle_attribute_vector(v1, v2, v3, depth_v1, depth_v2, depth_v3);

  for (tsi.reset(); tsi.next(); )
  {
    int y = tsi.scan_y();
    if (y < 0 || y >= static_cast<int>(img.height()))
      continue;
    int min_x = std::max(0, tsi.start_x());
    int max_x = std::min(static_cast<int>(img.width()) - 1, tsi.end_x());

    double new_i = Vd.y() * y + Vd.z();
    for (int x = min_x; x <= max_x; ++x)
    {
      double depth = new_i + Vd.x() * x;
      if (depth < depth_img(x, y))
      {
        depth_img(x, y) = depth;
        img(x, y) = value;
      }
    }
  }
}

}
}
}
#endif // KWIVER_ARROWS_CORE_RENDER_MESH_DEPTH_MAP_H
