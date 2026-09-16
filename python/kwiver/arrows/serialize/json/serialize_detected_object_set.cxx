// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_detected_object_set.h>
#include <python/kwiver/arrows/serialize/json/serialize_utils.txx>

#include <arrows/serialize/json/detected_object_set.h>
#include <viame/core_types/any.h>
#include <viame/core_types/detected_object_set.h>

namespace viame {

namespace python {

void
serialize_detected_object_set( py::module& m )
{
  m.def(
    "serialize_detected_object_set",
    &viame::python::arrows::json::serialize<
      viame::detected_object_set_sptr,
      viame::serialize::json::detected_object_set > );
  m.def(
    "deserialize_detected_object_set",
    &viame::python::arrows::json::deserialize<
      viame::detected_object_set_sptr,
      viame::serialize::json::detected_object_set > );
}

} // namespace python

} // namespace viame
