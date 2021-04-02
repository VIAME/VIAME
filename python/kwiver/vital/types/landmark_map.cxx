// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/landmark_map.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kv::simple_landmark_map s_landmark_map;
typedef std::map< kv::landmark_id_t, kv::landmark_sptr > map_landmark_t;
using namespace kwiver::vital;

PYBIND11_MODULE(landmark_map, m)
{
  py::bind_map< map_landmark_t >(m, "LandmarkDict");

  py::class_< landmark_map, std::shared_ptr< landmark_map > >(m, "LandmarkMap")
  .def("size", &landmark_map::size)
  .def("landmarks", &landmark_map::landmarks, py::return_value_policy::reference)

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

  py::class_< s_landmark_map, landmark_map, std::shared_ptr< s_landmark_map > >(m, "SimpleLandmarkMap")
  .def(py::init<>())
  .def(py::init< map_landmark_t >());
}
