/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

#include <vital/types/landmark.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

void landmark_map(py::module &m) {
    py::bind_map<map_landmark_t>(m, "LandmarkDict");

    py::class_<landmark_map_t, std::shared_ptr<landmark_map_t>>(m, "BaseLandmarkMap")
    .def("size", &landmark_map_t::size)
    .def("landmarks", &landmark_map_t::landmarks, py::return_value_policy::reference_internal)

    .def("__repr__", [](py::object& self) -> std::string {
        auto locals = py::dict(py::arg("self")=self);
        py::exec(R"(
            classname = self.__class__.__name__
            retval = '<%s at %s>' % (classname, hex(id(self)))
            )", py::globals(), locals);
        return locals["retval"].cast<std::string>();
    })

    .def("__str__", [](py::object& self) -> std::string {
        auto locals = py::dict(py::arg("self")=self);
        py::exec(R"(
            classname = self.__class__.__name__
            retval = '<%s>' % (classname)
            )", py::globals(), locals);
        return locals["retval"].cast<std::string>();
    });


    py::class_<s_landmark_map_t, landmark_map_t, std::shared_ptr<s_landmark_map_t>>(m, "LandmarkMap")
    .def(py::init<>())
    .def(py::init<map_landmark_t>());
}
