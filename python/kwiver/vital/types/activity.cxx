/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <python/kwiver/vital/types/activity.h>
#include <vital/vital_types.h>
#include <vital/types/activity.h>
#include <vital/types/activity_type.h>
#include <vital/types/object_track_set.h>

#include <pybind11/stl.h>

namespace py = pybind11;

void activity( py::module& m )
{
  py::class_<kwiver::vital::activity,
             std::shared_ptr<kwiver::vital::activity>>(m, "Activity")
    .def(py::init<>())
    .def(py::init<kwiver::vital::activity_id_t,
                  kwiver::vital::activity_label_t,
                  kwiver::vital::activity_confidence_t,
                  kwiver::vital::activity_type_sptr,
                  kwiver::vital::timestamp,
                  kwiver::vital::timestamp,
                  kwiver::vital::object_track_set_sptr>(),
          py::arg("activity_id") = -1,
          py::arg("activity_label") = kwiver::vital::UNDEFINED_ACTIVITY,
          py::arg("activity_confidence") = -1.0,
          py::arg("activity_type") =
           std::make_shared<kwiver::vital::activity_type>(kwiver::vital::UNDEFINED_ACTIVITY,
                                                        -1.0),
          py::arg("start_time") = kwiver::vital::timestamp(-1, -1),
          py::arg("end_time") = kwiver::vital::timestamp(-1, -1),
          py::arg("participants") =
           std::make_shared<kwiver::vital::object_track_set>() )
    .def("id", &kwiver::vital::activity::id)
    .def("set_id", &kwiver::vital::activity::set_id)
    .def("label",  &kwiver::vital::activity::label)
    .def("set_label", &kwiver::vital::activity::set_label)
    .def("activity_type", &kwiver::vital::activity::activity_type)
    .def("set_activity_type", &kwiver::vital::activity::set_activity_type)
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
