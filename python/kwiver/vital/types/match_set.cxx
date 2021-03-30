// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
