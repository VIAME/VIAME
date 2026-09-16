// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_track.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/track.h>
#include <viame/core_types/any.h>
#include <viame/core_types/track.h>

namespace viame {

namespace python {

void
serialize_track( py::module& m )
{
  m.def(
    "serialize_track",
    &viame::python::arrows::json::serialize<
      viame::track_sptr,
      viame::serialize::json::track > );
  m.def(
    "deserialize_track",
    &viame::python::arrows::json::deserialize<
      viame::track_sptr,
      viame::serialize::json::track > );
}

} // namespace python

} // namespace viame
