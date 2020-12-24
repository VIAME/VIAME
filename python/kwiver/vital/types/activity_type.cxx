// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/activity_type.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(activity_type, m)
{
  py::class_<kwiver::vital::activity_type, std::shared_ptr<kwiver::vital::activity_type>>(m, "ActivityType")
  .def(py::init<>())
  .def(py::init<std::vector<std::string>, std::vector<double>>())
  .def(py::init<std::string, double>())

  .def("has_class_name", &kwiver::vital::activity_type::has_class_name,
    py::arg("class_name"))
  .def("score", &kwiver::vital::activity_type::score,
    py::arg("class_name"))
  .def("get_most_likely_class", [](std::shared_ptr<kwiver::vital::activity_type> self)
    {
      std::string max_name;
      double max_score;
      self->get_most_likely(max_name, max_score);
      return max_name;
    })
  .def("get_most_likely_score", [](std::shared_ptr<kwiver::vital::activity_type> self)
    {
      std::string max_name;
      double max_score;
      self->get_most_likely(max_name, max_score);
      return max_score;
    })
  .def("set_score", &kwiver::vital::activity_type::set_score,
    py::arg("class_name"), py::arg("score"))
  .def("delete_score", &kwiver::vital::activity_type::delete_score,
    py::arg("class_name"))
  .def("class_names", &kwiver::vital::activity_type::class_names,
    py::arg("threshold")=kwiver::vital::activity_type::INVALID_SCORE)
  .def_static("all_class_names", &kwiver::vital::activity_type::all_class_names)
  .def("__len__", &kwiver::vital::activity_type::size)
  .def("__iter__", [](kwiver::vital::activity_type& self){return py::make_iterator(self.cbegin(),self.cend());},
          py::keep_alive<0,1>())
  .def("__repr__", [](py::object& self)->std::string
    {
      auto info = py::dict(py::arg("self")=self);
      py::exec(R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
      )", py::globals(),info);
      return info["retval"].cast<std::string>();
    })
  .def("__nice__", [](kwiver::vital::activity_type& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        retval = 'size={}'.format(len(self))
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  .def("__str__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s)>' % (classname, devnice)
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  ;
}
