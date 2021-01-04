// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_track.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/track.h>
#include <vital/types/track.h>
#include <vital/any.h>

namespace kwiver {
namespace arrows {
namespace python {
void serialize_track(py::module &m)
{
  m.def("serialize_track",
        &kwiver::python::arrows::json::serialize<
                          kwiver::vital::track_sptr,
                          kwiver::arrows::serialize::json::track > );
  m.def("deserialize_track",
        &kwiver::python::arrows::json::deserialize<
                          kwiver::vital::track_sptr,
                          kwiver::arrows::serialize::json::track> );
}
}
}
}
