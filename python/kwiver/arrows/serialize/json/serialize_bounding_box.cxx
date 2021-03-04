// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_bounding_box.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/bounding_box.h>
#include <vital/types/bounding_box.h>
#include <vital/any.h>
namespace kwiver {
namespace arrows {
namespace python {
void serialize_bounding_box(py::module &m)
{
  m.def("serialize_bounding_box",
        &kwiver::python::arrows::json::serialize<
                              kwiver::vital::bounding_box<double>,
                              kwiver::arrows::serialize::json::bounding_box > );
  m.def("deserialize_bounding_box",
        &kwiver::python::arrows::json::deserialize<
                              kwiver::vital::bounding_box<double>,
                              kwiver::arrows::serialize::json::bounding_box > );
}
}
}
}
