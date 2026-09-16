// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "render_mesh_depth_map.h"

#include <arrows/core/mesh_intersect.h>
#include <arrows/core/mesh_operations.h>
#include <arrows/core/render_mesh_depth_map.h>
#include <viame/core_types/camera.h>
#include <viame/core_types/mesh.h>

py::tuple
run_mesh_closest_point(
  viame::point_3d const& p,
  std::shared_ptr< viame::mesh > const mesh,
  viame::point_3d& closest_point )
{
  double u, v;
  auto triangle_idx = viame::core::mesh_closest_point(
    p, *mesh, closest_point, u, v );
  return py::make_tuple( triangle_idx, u, v );
}

py::tuple
run_mesh_intersect(
  viame::point_3d const& p,
  viame::vector_3d const& direction,
  std::shared_ptr< viame::mesh > const mesh )
{
  double dist, u, v;
  auto triangle_idx = viame::core::mesh_intersect(
    p, direction, *mesh, dist, u, v );
  return py::make_tuple( triangle_idx, dist, u, v );
}

void
render_mesh_depth_map( py::module& m )
{
  m.def(
    "render_mesh_depth_map",
    [](std::shared_ptr< viame::mesh > const mesh,
       std::shared_ptr< viame::simple_camera_perspective > const cam){
      return viame::core::render_mesh_depth_map( mesh, cam );
    } );
  m.def(
    "mesh_triangulate",
    [](std::shared_ptr< viame::mesh > const mesh){
      viame::core::mesh_triangulate( *mesh );
      return mesh;
    } );
  m.def(
    "clip_mesh",
    [](std::shared_ptr< viame::mesh > const mesh,
       std::shared_ptr< viame::simple_camera_perspective > const cam){
      return viame::core::clip_mesh( *mesh, *cam );
    } );
  m.def( "mesh_closest_point", &run_mesh_closest_point );
  m.def( "mesh_intersect", &run_mesh_intersect );
}
