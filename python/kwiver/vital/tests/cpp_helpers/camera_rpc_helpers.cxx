// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera_rpc.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kwiver::vital::camera_rpc crpc;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( camera_rpc_helpers, m )
{
  m.def( "call_clone", [] ( const kv::camera_rpc &self)
  {
    return self.clone();
  });

  m.def( "call_rpc_coeffs", [] ( const kv::camera_rpc &self )
  {
    return self.rpc_coeffs();
  });

  m.def( "call_world_scale", [] ( const kv::camera_rpc &self )
  {
    return self.world_scale();
  });

  m.def( "call_world_offset", [] ( const kv::camera_rpc &self )
  {
    return self.world_offset();
  });

  m.def( "call_image_scale", [] ( const kv::camera_rpc &self )
  {
    return self.image_scale();
  });

  m.def( "call_image_offset", [] ( const kv::camera_rpc &self )
  {
    return self.image_offset();
  });

  m.def( "call_image_width", [] ( const kv::camera_rpc &self )
  {
    return self.image_width();
  });

  m.def( "call_image_height", [] ( const kv::camera_rpc &self )
  {
    return self.image_height();
  });

  m.def( "call_project", [] ( const kv::camera_rpc &self,
                               const kv::vector_3d &pt )
  {
    return self.project(pt);
  });

  m.def( "call_back_project", [] ( const kv::camera_rpc &self,
                                         kv::vector_2d &pt,
                                         double elev )
  {
    return self.back_project(pt, elev);
  });
}
