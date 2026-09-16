// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_timestamp.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/timestamp.h>
#include <viame/core_types/any.h>
#include <viame/core_types/timestamp.h>

namespace viame {

namespace python {

void
serialize_timestamp( py::module& m )
{
  m.def(
    "serialize_timestamp",
    &viame::python::arrows::json::serialize<
      viame::timestamp,
      viame::serialize::json::timestamp > );
  m.def(
    "deserialize_timestamp",
    &viame::python::arrows::json::deserialize<
      viame::timestamp,
      viame::serialize::json::timestamp > );
}

} // namespace python

} // namespace viame
