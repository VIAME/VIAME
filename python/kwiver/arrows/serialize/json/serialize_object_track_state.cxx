// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_object_track_state.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/object_track_state.h>
#include <viame/core_types/any.h>
#include <viame/core_types/object_track_set.h>

namespace viame {

namespace python {

void
serialize_object_track_state( py::module& m )
{
  m.def(
    "serialize_object_track_state",
    &viame::python::arrows::json::serialize<
      viame::object_track_state,
      viame::serialize::json::object_track_state > );
  m.def(
    "deserialize_object_track_state",
    &viame::python::arrows::json::deserialize<
      viame::object_track_state,
      viame::serialize::json::object_track_state > );
}

} // namespace python

} // namespace viame
