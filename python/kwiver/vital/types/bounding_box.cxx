// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#include <string.h>
#include <vital/types/bounding_box.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>

namespace py = pybind11;


template<typename T>
void bounding_box(py::module &m, const char * typestr)
{

  /*
   *

    Developer:
      python -c "from kwiver.vital.types import types"
      python -c "import kwiver.vital.types; help(kwiver.vital.types.BoundingBox)"
      python -c "import kwiver.vital.types; help(kwiver.vital.types.BoundingBox)"
      python -m xdoctest kwiver.vital.types BoundingBox --xdoc-dynamic

   *
   */
  typedef kwiver::vital::bounding_box<T> bbox;
  char pyclass_name[20];
  strcpy(pyclass_name, "BoundingBox");
  strcat(pyclass_name, typestr);

  py::class_<bbox, std::shared_ptr<bbox>>(m, pyclass_name, R"(
    Coordinate aligned bounding box.

    Example:
        >>> from kwiver.vital.types import *
        >>> bbox = BoundingBox(0, 10, 100, 50)
        >>> print(str(bbox))
        <BoundingBox(0.0, 10.0, 100.0, 50.0)>
        >>> print(bbox.area())
        4000.0

    )")
  .def(py::init<Eigen::Matrix<T,2,1>, Eigen::Matrix<T,2,1>>())
  .def(py::init<Eigen::Matrix<T,2,1>, T, T>())
  .def(py::init<T, T, T, T>(), py::doc(R"(
        Create a box from four coordinates

        Args:
            xmin (float):  min x coord
            ymin (float):  min y coord
            xmax (float):  max x coord
            ymax (float):  max y coord
        )"))
  .def(py::init<>(), py::doc(R"(
    Create a new default Bounding Box
    It is empty, and invalid.
    )"))
  .def("is_valid", &bbox::is_valid)
  .def("center", &bbox::center)
  .def("upper_left", &bbox::upper_left)
  .def("lower_right", &bbox::lower_right)
  .def("min_x", &bbox::min_x)
  .def("min_y", &bbox::min_y)
  .def("max_x", &bbox::max_x)
  .def("max_y", &bbox::max_y)
  .def("width", &bbox::width)
  .def("height", &bbox::height)
  .def("area", &bbox::area)
  .def("contains", &bbox::contains)

  .def("__nice__", [](bbox& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        retval = '{}, {}, {}, {}'.format(self.min_x(), self.min_y(),
                                         self.max_x(), self.max_y())
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  .def("__repr__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
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
  .def("__eq__", [](bbox self, bbox other) {return self == other;})
  .def("__ne__", [](bbox self, bbox other) {return self != other;})
  ;
}

PYBIND11_MODULE(bounding_box, m)
{
  bounding_box<double>(m, "D");
  bounding_box<float>(m, "F");
  bounding_box<int>(m, "I");
}
