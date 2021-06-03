// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "render_mesh_depth_map.h"

#include <arrows/core/render_mesh_depth_map.h>
#include <arrows/core/mesh_operations.h>
#include <vital/types/mesh.h>
#include <vital/types/camera.h>

void render_mesh_depth_map(py::module &m)
{
  m.def("render_mesh_depth_map",
        [](std::shared_ptr<kwiver::vital::mesh> const mesh,
           std::shared_ptr<kwiver::vital::simple_camera_perspective> const cam)
        {
          return kwiver::arrows::core::render_mesh_depth_map(mesh, cam);
        });
  m.def("mesh_triangulate",
        [](std::shared_ptr<kwiver::vital::mesh> const mesh)
        {
          kwiver::arrows::core::mesh_triangulate(*mesh);
          return mesh;
        });
  m.def("clip_mesh",
        [](std::shared_ptr<kwiver::vital::mesh> const mesh,
           std::shared_ptr<kwiver::vital::simple_camera_perspective> const cam)
        {
          return kwiver::arrows::core::clip_mesh(*mesh, *cam);
        });
}
