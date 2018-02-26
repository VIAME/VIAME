/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include <vital/types/timestamp.h>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

namespace py = pybind11;

PYBIND11_MODULE(timestamp, m)
{
  py::class_<kwiver::vital::timestamp, std::shared_ptr<kwiver::vital::timestamp>>(m, "Timestamp")
  .def(py::init<>())
  .def(py::init<int64_t, int64_t>())
  .def("is_valid", &kwiver::vital::timestamp::is_valid)
  .def("has_valid_time", &kwiver::vital::timestamp::has_valid_time)
  .def("has_valid_frame", &kwiver::vital::timestamp::has_valid_frame)
  .def("get_time_usec", &kwiver::vital::timestamp::get_time_usec) // We can't treat time as a property, because of multiple accessors/mutators
  .def("set_time_usec", &kwiver::vital::timestamp::set_time_usec,
    py::arg("time"))
  .def("get_time_seconds", &kwiver::vital::timestamp::get_time_seconds)
  .def("set_time_seconds", &kwiver::vital::timestamp::set_time_seconds,
    py::arg("time"))
  .def("get_frame", &kwiver::vital::timestamp::get_frame) // For consistency, let's not treat frame as a property either
  .def("set_frame", &kwiver::vital::timestamp::set_frame,
    py::arg("frame"))
  .def("set_invaild", &kwiver::vital::timestamp::set_invalid)
  .def("set_time_domain_index", &kwiver::vital::timestamp::set_time_domain_index,
    py::arg("domain"))
  .def("get_time_domain_index", &kwiver::vital::timestamp::get_time_domain_index)
  .def("__str__", &kwiver::vital::timestamp::pretty_print)
  .def(py::self == py::self)
  .def(py::self != py::self)
  .def(py::self <= py::self)
  .def(py::self >= py::self)
  .def(py::self < py::self)
  .def(py::self > py::self)
  ;
}
