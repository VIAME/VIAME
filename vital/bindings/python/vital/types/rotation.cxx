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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rotation_class.cxx"

namespace py = pybind11;

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
