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

#include <vital/types/descriptor_set.h>

#include <memory>

namespace py = pybind11;

typedef kwiver::vital::descriptor_set desc_set;
typedef kwiver::vital::simple_descriptor_set s_desc_set;

std::shared_ptr<s_desc_set>
new_desc_set()
{
  return std::make_shared<s_desc_set>();
}

std::shared_ptr<s_desc_set>
new_desc_set1(py::list py_list)
{
  std::vector<std::shared_ptr<kwiver::vital::descriptor>> desc_list;
  for(auto py_desc : py_list)
  {
    desc_list.push_back(py::cast<std::shared_ptr<kwiver::vital::descriptor>>(py_desc));
  }
  return std::make_shared<s_desc_set>(desc_list);
}

PYBIND11_MODULE(descriptor_set, m)
{
  py::class_<desc_set, std::shared_ptr<desc_set>>(m, "BaseDescriptorSet");

  py::class_<s_desc_set, desc_set, std::shared_ptr<s_desc_set>>(m, "DescriptorSet")
  .def(py::init(&new_desc_set))
  .def(py::init(&new_desc_set1),
    py::arg("list"))
  .def("descriptors", &s_desc_set::descriptors)
  .def("size", &s_desc_set::size)
  .def("__len__", &s_desc_set::size)
  ;

}
