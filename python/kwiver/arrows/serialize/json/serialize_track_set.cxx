// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_track_set.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/track_set.h>
#include <viame/core_types/any.h>
#include <viame/core_types/track_set.h>

namespace viame {

namespace python {

void
serialize_track_set( py::module& m )
{
  m.def(
    "serialize_track_set",
    &viame::python::arrows::json::serialize<
      viame::track_set_sptr,
      viame::serialize::json::track_set > );
  m.def(
    "deserialize_track_set",
    &viame::python::arrows::json::deserialize<
      viame::track_set_sptr,
      viame::serialize::json::track_set > );
}

} // namespace python

} // namespace viame
