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

#include <vital/types/category_hierarchy.h>

namespace py = pybind11;

PYBIND11_MODULE(category_hierarchy, m)
{
  py::class_<kwiver::vital::category_hierarchy,
    std::shared_ptr<kwiver::vital::category_hierarchy>>(m, "CategoryHierarchy")
  .def(py::init<>())
  .def(py::init<std::vector<std::string>>())
  .def(py::init<std::vector<std::string>,std::vector<std::string>>())
  .def(py::init<std::vector<std::string>,std::vector<std::string>,std::vector<int>>())

  .def("add_class", &kwiver::vital::category_hierarchy::add_class,
    py::arg("class_name"), py::arg("class_parent"), py::arg("class_id"))
  .def("has_class_id", &kwiver::vital::category_hierarchy::has_class_name,
    py::arg("class_name"))
  .def("get_class_name", &kwiver::vital::category_hierarchy::get_class_name,
    py::arg("class_name"))
  .def("get_class_id", &kwiver::vital::category_hierarchy::get_class_id,
    py::arg("class_name"))
  .def("get_class_parents", &kwiver::vital::category_hierarchy::get_class_parents,
    py::arg("class_name"))
  .def("all_class_names", &kwiver::vital::category_hierarchy::all_class_names)
  ;
}
