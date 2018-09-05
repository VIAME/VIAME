/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include <pybind11/stl.h>

#include <vital/types/detected_object_type.h>

namespace py = pybind11;

PYBIND11_MODULE(detected_object_type, m)
{
  py::class_<kwiver::vital::detected_object_type, std::shared_ptr<kwiver::vital::detected_object_type>>(m, "DetectedObjectType")
  .def(py::init<>())
  .def(py::init<std::vector<std::string>, std::vector<double>>())
  .def(py::init<std::string, double>())

  .def("has_class_name", &kwiver::vital::detected_object_type::has_class_name,
    py::arg("class_name"))
  .def("score", &kwiver::vital::detected_object_type::score,
    py::arg("class_name"))
  .def("get_most_likely_class", [](std::shared_ptr<kwiver::vital::detected_object_type> self)
    {
      std::string max_name;
      double max_score;
      self->get_most_likely(max_name, max_score);
      return max_name;
    })
  .def("get_most_likely_score", [](std::shared_ptr<kwiver::vital::detected_object_type> self)
    {
      std::string max_name;
      double max_score;
      self->get_most_likely(max_name, max_score);
      return max_score;
    })
  .def("set_score", &kwiver::vital::detected_object_type::set_score,
    py::arg("class_name"), py::arg("score"))
  .def("delete_score", &kwiver::vital::detected_object_type::delete_score,
    py::arg("class_name"))
  .def("class_names", &kwiver::vital::detected_object_type::class_names,
    py::arg("threshold")=kwiver::vital::detected_object_type::INVALID_SCORE)
  .def_static("all_class_names", &kwiver::vital::detected_object_type::all_class_names)
  ;
}
