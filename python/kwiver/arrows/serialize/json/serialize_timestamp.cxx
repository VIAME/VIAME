// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_timestamp.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/timestamp.h>
#include <vital/types/timestamp.h>
#include <vital/any.h>

namespace kwiver {
namespace arrows {
namespace python {
void serialize_timestamp(py::module &m)
{
  m.def("serialize_timestamp",
        &kwiver::python::arrows::json::serialize<
                              kwiver::vital::timestamp,
                              kwiver::arrows::serialize::json::timestamp > );
  m.def("deserialize_timestamp",
        &kwiver::python::arrows::json::deserialize<
                              kwiver::vital::timestamp,
                              kwiver::arrows::serialize::json::timestamp > );
}
}
}
}
