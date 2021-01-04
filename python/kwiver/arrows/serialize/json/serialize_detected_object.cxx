// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_detected_object.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/detected_object.h>
#include <vital/types/detected_object.h>
#include <vital/any.h>

namespace kwiver {
namespace arrows {
namespace python {
void serialize_detected_object(py::module &m)
{
  m.def("serialize_detected_object",
        &kwiver::python::arrows::json::serialize<
                          kwiver::vital::detected_object_sptr,
                          kwiver::arrows::serialize::json::detected_object > );
  m.def("deserialize_detected_object",
        &kwiver::python::arrows::json::deserialize<
                          kwiver::vital::detected_object_sptr,
                          kwiver::arrows::serialize::json::detected_object > );
}
}
}
}
