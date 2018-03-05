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

#include "descriptor_class.cxx"

namespace py = pybind11;

PYBIND11_MODULE(descriptor, m)
{
  py::class_<PyDescriptorBase, std::shared_ptr<PyDescriptorBase>>(m, "Descriptor")
  .def(py::init(&new_descriptor),
    py::arg("size")=0, py::arg("ctype")='d')
  .def("sum", &PyDescriptorBase::sum)
  .def("todoublearray", &PyDescriptorBase::as_double)
  .def("tobytearray", &PyDescriptorBase::as_bytes)
  .def("__eq__", [](PyDescriptorBase &self, PyDescriptorBase &other) { return self.as_double() == other.as_double(); })
  .def("__ne__", [](PyDescriptorBase &self, PyDescriptorBase &other) { return self.as_double() != other.as_double(); })
  .def("__setitem__", &PyDescriptorBase::set_slice,
    py::arg("slice"), py::arg("value"))
  .def("__getitem__", &PyDescriptorBase::get_slice,
    py::arg("slice"))
  .def_property_readonly("size", &PyDescriptorBase::get_size)
  .def_property_readonly("nbytes", &PyDescriptorBase::get_num_bytes)
  ;

  py::class_<PyDescriptorD, PyDescriptorBase, std::shared_ptr<PyDescriptorD>>(m, "DescriptorD");
  py::class_<PyDescriptorF, PyDescriptorBase, std::shared_ptr<PyDescriptorF>>(m, "DescriptorF");
}
