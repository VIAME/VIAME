// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_object_track_state.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/object_track_state.h>
#include <vital/types/object_track_set.h>
#include <vital/any.h>

namespace kwiver {
namespace arrows {
namespace python {
void serialize_object_track_state(py::module &m)
{
  m.def("serialize_object_track_state",
        &kwiver::python::arrows::json::serialize<
                          kwiver::vital::object_track_state,
                          kwiver::arrows::serialize::json::object_track_state > );
  m.def("deserialize_object_track_state",
        &kwiver::python::arrows::json::deserialize<
                          kwiver::vital::object_track_state,
                          kwiver::arrows::serialize::json::object_track_state> );
}
}
}
}
