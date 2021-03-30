// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera_perspective_map.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <python/kwiver/vital/util/pybind11.h>
#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( camera_perspective_map_helpers, m )
{
  m.def( "call_size", [] ( const kv::camera_map_of_< kv::camera_perspective > &self )
  {
    return self.size();
  });

  m.def( "call_cameras", [] ( const kv::camera_perspective_map &self )
  {
    return self.cameras();
  });

  m.def( "call_get_frame_ids", [] ( const kv::camera_map_of_< kv::camera_perspective > &self )
  {
    return self.get_frame_ids();
  });
}
