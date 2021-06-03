// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera_perspective.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kwiver::vital::camera_perspective cam_p;
// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( camera_perspective_helpers, m )
{
  m.def( "call_clone", [] ( const kv::camera_perspective &self)
  {
    return self.clone();
  });

  m.def( "call_center", [] ( const kv::camera_perspective &self )
  {
    return self.center();
  });

  m.def( "call_translation", [] ( const kv::camera_perspective &self )
  {
    return self.translation();
  });

  m.def( "call_center_covar", [] ( const kv::camera_perspective &self )
  {
    return self.center_covar();
  });

  m.def( "call_rotation", [] ( const kv::camera_perspective &self )
  {
    return self.rotation();
  });

  m.def( "call_intrinsics", [] ( const kv::camera_perspective &self )
  {
    return self.intrinsics();
  });

  m.def( "call_image_width", [] ( const kv::camera_perspective &self )
  {
    return self.image_width();
  });

  m.def( "call_image_height", [] ( const kv::camera_perspective &self )
  {
    return self.image_height();
  });

  m.def( "call_clone_look_at", [] ( const kv::camera_perspective &self,
                               const kv::vector_3d &stare_pt,
                               const kv::vector_3d &up_direc )
  {
    return self.clone_look_at(stare_pt,up_direc);
  });

  m.def( "call_as_matrix", [] ( const kv::camera_perspective &self )
  {
    return self.as_matrix();
  });

  m.def( "call_project", [] ( const kv::camera_perspective &self, kv::vector_3d &pt )
  {
    return self.project(pt);
  });

  m.def( "call_depth", [] ( const kv::camera_perspective &self, kv::vector_3d &pt )
  {
    return self.depth(pt);
  });
}
