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

#include <python/kwiver/vital/types/activity_type.h>
#include <vital/vital_types.h>
#include <vital/types/activity_type.h>

#include <pybind11/stl.h>

namespace py = pybind11;

void activity_type( py::module& m )
{
  py::class_<kwiver::vital::activity_type,
             std::shared_ptr<kwiver::vital::activity_type>>(m, "ActivityType")
    .def(py::init<>())
    .def(py::init<std::vector<kwiver::vital::activity_label_t>,
                  std::vector<kwiver::vital::activity_confidence_t>>())
    .def(py::init<kwiver::vital::activity_label_t,
                  kwiver::vital::activity_confidence_t>())
    .def("has_class_name", &kwiver::vital::activity_type::has_class_name)
    .def("score", &kwiver::vital::activity_type::score)
    .def("get_most_likely_class",
          &kwiver::vital::activity_type::get_most_likely_class)
    .def("get_most_likely_class_and_score",
          &kwiver::vital::activity_type::get_most_likely_class_and_score)
    .def("set_score", &kwiver::vital::activity_type::set_score)
    .def("delete_score", &kwiver::vital::activity_type::delete_score)
    .def("class_names", &kwiver::vital::activity_type::class_names)
    .def("size", &kwiver::vital::activity_type::size)
    .def("all_class_names", &kwiver::vital::activity_type::all_class_names);
}
