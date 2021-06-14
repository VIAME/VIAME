// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/detected_object_set.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kwiver::vital::detected_object_set det_obj_set;
// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( det_obj_set_helpers, m )
{
  m.def( "call_size", [] (const det_obj_set &obj)
  {
    return obj.size();
  });

  m.def( "call_empty", [] (const det_obj_set &obj)
  {
    return obj.empty();
  });
  m.def( "call_at", [] (const det_obj_set &obj, int pos)
  {
    return obj.at(pos);
  });
}
