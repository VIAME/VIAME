// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( camera_helpers, m )
{
  m.def( "call_clone", [] ( const kv::camera& cam )
  {
    return cam.clone();
  });

  m.def( "call_project", [] ( const kv::camera& cam, const kv::vector_3d& pt )
  {
    return cam.project(pt);
  });

  m.def( "call_image_width", [] ( const kv::camera& cam )
  {
    return cam.image_width();
  });

  m.def( "call_image_height", [] ( const kv::camera& cam )
  {
    return cam.image_height();
  });
}
