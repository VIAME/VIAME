// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rotation_class.cxx"

namespace py = pybind11;

using namespace kwiver::vital::python;

PYBIND11_MODULE(rotation, m)
{
  py::class_<PyRotation, std::shared_ptr<PyRotation> >(m, "Rotation")
  .def(py::init<char>(),
       py::arg("type")='d')
  .def_static("from_axis_angle", &PyRotation::from_axis_angle,
       py::arg("axis"), py::arg("angle"), py::arg("ctype")='d')
  .def_static("from_matrix", &PyRotation::from_matrix,
       py::arg("matrix"), py::arg("ctype")='d')
  .def_static("from_quaternion", &PyRotation::from_quaternion,
       py::arg("quaternion"), py::arg("ctype")='d')
  .def_static("from_rodrigues", &PyRotation::from_rodrigues,
       py::arg("rodrigues"), py::arg("ctype")='d')
  .def_static("from_ypr", &PyRotation::from_ypr,
       py::arg("yaw"), py::arg("pitch"), py::arg("roll"), py::arg("ctype")='d')
  .def_static("interpolate", &PyRotation::interpolate,
       py::arg("A"), py::arg("B"), py::arg("f"))
  .def_static("interpolated_rotations", &PyRotation::interpolated_rotations,
       py::arg("A"), py::arg("B"), py::arg("n"))
  .def("__eq__", [](const std::shared_ptr<PyRotation> self,
                    const std::shared_ptr<PyRotation> other)
                    {
                      if(self->getType() == 'd')
                      {
                        if(other->getType() == 'f') // it's okay if they're different types
                        {
                          other->convert_to_d();
                          other->setType('f'); // revert the type after populating rotation
                        }
                        return self->getRotD() == other->getRotD();
                      }
                      else if(self->getType() == 'f')
                      {
                        if(other->getType() == 'd') // it's okay if they're different types
                        {
                          other->convert_to_f();
                          other->setType('d'); // revert the type after populating rotation
                        }
                        return self->getRotD() == other->getRotD();
                      }
                      return false;
                    })
  .def("__ne__", [](const std::shared_ptr<PyRotation> self,
                    const std::shared_ptr<PyRotation> other)
                    {
                      if(self->getType() == 'd')
                      {
                        if(other->getType() == 'f') // it's okay if they're different types
                        {
                          other->convert_to_d();
                          other->setType('f'); // revert the type after populating rotation
                        }
                        return self->getRotD() != other->getRotD();
                      }
                      else if(self->getType() == 'f')
                      {
                        if(other->getType() == 'd') // it's okay if they're different types
                        {
                          other->convert_to_f();
                          other->setType('d'); // revert the type after populating rotation
                        }
                        return self->getRotD() != other->getRotD();
                      }
                      return false;
                    })
  .def("angle", &PyRotation::angle)
  .def("angle_from", &PyRotation::angle_from,
       py::arg("other"))
  .def("axis", &PyRotation::axis)
  .def("compose", &PyRotation::compose,
       py::arg("other"))
  .def("inverse", &PyRotation::inverse)
  .def("matrix", &PyRotation::matrix)
  .def("quaternion", &PyRotation::quaternion)
  .def("rodrigues", &PyRotation::rodrigues)
  .def("rotate_vector", &PyRotation::rotate_vector,
       py::arg("vector"))
  .def("yaw_pitch_roll", &PyRotation::get_yaw_pitch_roll)
  .def("__mul__", [](std::shared_ptr<PyRotation> self, std::shared_ptr<PyRotation> rhs)
                    {
                        PyRotation composed = self->compose(rhs);
                        return py::cast<PyRotation>(composed);
                    })
  .def("__mul__", [](std::shared_ptr<PyRotation> self, py::object rhs)
                    {
                      return self->rotate_vector(rhs);
                    })
  .def_property_readonly("_ctype", &PyRotation::getType)
  ;
}
