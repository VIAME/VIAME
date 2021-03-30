// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/transform_2d.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these transform_2d methods can be overriden in C++
PYBIND11_MODULE( transform_2d_helpers, m )
{
  m.def( "call_clone", [] ( const kv::transform_2d& t )
  {
    return t.clone();
  });

  m.def( "call_map", [] ( const kv::transform_2d& t, const kv::vector_2d& p )
  {
    return t.map(p);
  });

  m.def( "call_inverse", [] (const kv::transform_2d &t)
  {
    return t.inverse();
  });
}
