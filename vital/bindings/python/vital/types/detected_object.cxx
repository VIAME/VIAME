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

#include <vital/types/detected_object.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

typedef kwiver::vital::detected_object det_obj;

PYBIND11_MODULE(detected_object, m)
{

  py::class_<det_obj, std::shared_ptr<det_obj>>(m, "DetectedObject")
  .def(py::init<kwiver::vital::bounding_box<double>, double, kwiver::vital::detected_object_type_sptr>(),
    py::arg("bbox"), py::arg("confidence")=1.0, py::arg("classifications")=kwiver::vital::detected_object_type_sptr())
  .def("bounding_box", &det_obj::bounding_box)
  .def("set_bounding_box", &det_obj::set_bounding_box,
    py::arg("bbox"))
  .def("confidence", &det_obj::confidence)
  .def("set_confidence", &det_obj::set_confidence,
    py::arg("d"))
  .def("type", &det_obj::type)
  .def("set_type", &det_obj::set_type,
    py::arg("c"))
  ;
}
