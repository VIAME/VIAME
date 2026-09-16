// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_activity.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/activity.h>
#include <viame/core_types/any.h>
#include <viame/core_types/activity.h>

namespace viame {

namespace python {

void
serialize_activity( py::module& m )
{
  m.def(
    "serialize_activity",
    &viame::python::arrows::json::serialize<
      viame::activity,
      viame::serialize::json::activity > );
  m.def(
    "deserialize_activity",
    &viame::python::arrows::json::deserialize<
      viame::activity,
      viame::serialize::json::activity > );
}

} // namespace python

} // namespace viame
