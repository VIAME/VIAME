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
#include <pybind11/pybind11.h>
#include <vital/types/match_set.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kwiver::vital::match match_t;
typedef kwiver::vital::match_set match_set_t;
typedef kwiver::vital::simple_match_set s_match_set_t;

PYBIND11_MODULE(match_set, m)
{
  py::bind_vector<std::vector<match_t>>(m, "MatchVector");
  py::class_< match_set_t, std::shared_ptr<match_set_t> >(m, "BaseMatchSet")
  .def("size", &match_set_t::size)
  .def("matches", &match_set_t::matches, py::return_value_policy::reference_internal)

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


  py::class_<s_match_set_t, match_set_t, std::shared_ptr<s_match_set_t>>(m, "MatchSet")
  .def(py::init<>())
  .def(py::init<const std::vector<match_t>& >());

}
