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

#include <vital/types/detected_object_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef kwiver::vital::detected_object_set det_obj_set;

PYBIND11_MODULE(detected_object_set, m)
{
  py::class_<det_obj_set, std::shared_ptr<det_obj_set>>(m, "DetectedObjectSet")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::detected_object>>>())
  .def("add", [](det_obj_set &self, py::object object)
    {
      try
      {
        auto det_obj = object.cast<std::shared_ptr<kwiver::vital::detected_object>>();
        return self.add(det_obj);
      }
      catch(...){};
      auto det_obj = object.cast<std::shared_ptr<det_obj_set>>();
      return self.add(det_obj);
    })
  .def("size", &det_obj_set::size)
  .def("select", [](det_obj_set &self, double threshold, py::object class_name)
    {
      if(class_name.is(py::none()))
      {
        return self.select(threshold);
      }
      return self.select(class_name.cast<std::string>(), threshold);
    },
    py::arg("threshold")=kwiver::vital::detected_object_type::INVALID_SCORE, py::arg("class_name")=py::none())
  .def("__getitem__", [](det_obj_set &self, size_t idx)
    {
      return self.at(idx);
    })
  ;
}
