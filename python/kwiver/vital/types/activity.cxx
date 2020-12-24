// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/vital_types.h>
#include <vital/types/activity.h>
#include <vital/types/activity_type.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(activity, m)
{
  py::class_<kwiver::vital::activity,
             std::shared_ptr<kwiver::vital::activity>>(m, "Activity")
    .def(py::init<>())
    .def(py::init<kwiver::vital::activity_id_t,
                  kwiver::vital::activity_label_t,
                  double,
                  kwiver::vital::activity_type_sptr,
                  kwiver::vital::timestamp,
                  kwiver::vital::timestamp,
                  kwiver::vital::object_track_set_sptr>(),
          py::arg("id") = -1,
          py::arg("label") = kwiver::vital::UNDEFINED_ACTIVITY,
          py::arg("confidence") = -1.0,
          py::arg("classifications") = nullptr,
          py::arg("start_time") = kwiver::vital::timestamp(-1, -1),
          py::arg("end_time") = kwiver::vital::timestamp(-1, -1),
          py::arg("participants") =
           std::make_shared<kwiver::vital::object_track_set>() )
    .def("id", &kwiver::vital::activity::id)
    .def("set_id", &kwiver::vital::activity::set_id)
    .def("label",  &kwiver::vital::activity::label)
    .def("set_label", &kwiver::vital::activity::set_label)
    .def("type", &kwiver::vital::activity::type)
    .def("set_type", &kwiver::vital::activity::set_type)
    .def("confidence", &kwiver::vital::activity::confidence)
    .def("set_confidence", &kwiver::vital::activity::set_confidence)
    .def("start_time", &kwiver::vital::activity::start)
    .def("set_start_time", &kwiver::vital::activity::set_start)
    .def("end_time", &kwiver::vital::activity::end)
    .def("set_end_time", &kwiver::vital::activity::set_end)
    .def("duration", &kwiver::vital::activity::duration)
    .def("end", &kwiver::vital::activity::participants)
    .def("set_end", &kwiver::vital::activity::set_participants);
}
