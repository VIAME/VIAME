// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/vital_types.h>
#include <vital/types/activity.h>
#include <vital/types/activity_type.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE(activity, m)
{
  py::class_<kv::activity,
             std::shared_ptr<kv::activity>>(m, "Activity")
    .def(py::init<>())
    .def(py::init<kv::activity_id_t,
                  kv::activity_label_t,
                  double,
                  kv::activity_type_sptr,
                  kv::timestamp,
                  kv::timestamp,
                  kv::object_track_set_sptr>(),
          py::arg("activity_id"),
          py::arg("activity_label") = kv::UNDEFINED_ACTIVITY,
          py::arg("activity_confidence") = -1.0,
          py::arg("activity_type") = nullptr,
          py::arg("start_time") = kv::timestamp(-1, -1),
          py::arg("end_time") = kv::timestamp(-1, -1),
          py::arg("participants") = nullptr)
    .def_property("id", &kv::activity::id,
                        &kv::activity::set_id)
    .def_property("label", &kv::activity::label,
                           &kv::activity::set_label)
    .def_property("activity_type", &kv::activity::type,
                                   &kv::activity::set_type)
    .def_property("confidence", &kv::activity::confidence,
                                &kv::activity::set_confidence)
    .def_property("start_time", &kv::activity::start,
                                &kv::activity::set_start)
    .def_property("end_time", &kv::activity::end,
                              &kv::activity::set_end)
    .def_property("participants", &kv::activity::participants,
                                  &kv::activity::set_participants)
    .def_property_readonly("duration", &kv::activity::duration)
    ;
}
