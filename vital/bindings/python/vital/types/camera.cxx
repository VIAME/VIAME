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

#include <vital/io/camera_io.h>
#include <vital/types/camera_perspective.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "covariance_class.cxx"
#include "eigen_class.cxx"
#include "rotation_class.cxx"

namespace py = pybind11;

bool
camera_eq(std::shared_ptr<kwiver::vital::simple_camera_perspective> self,
          std::shared_ptr<kwiver::vital::simple_camera_perspective> other)
{
  return ((self->get_center() - other->get_center()).isMuchSmallerThan(0.001) &&
          (self->get_center_covar().matrix() == other->get_center_covar().matrix()) &&
          (self->get_rotation().matrix() == other->get_rotation().matrix()));
}

bool
camera_ne(std::shared_ptr<kwiver::vital::simple_camera_perspective> self,
          std::shared_ptr<kwiver::vital::simple_camera_perspective> other)
{
  return !camera_eq(self, other);
}

PYBIND11_MODULE(camera, m)
{

  py::class_<kwiver::vital::simple_camera_perspective,
             std::shared_ptr<kwiver::vital::simple_camera_perspective> >(m, "Camera")
  .def(py::init<>())
  .def(py::init([](EigenArray &center)
                  {
                    Eigen::Vector3d vec(center.getMatrixD().data());
                    kwiver::vital::simple_camera_perspective ret_cam(vec, kwiver::vital::rotation_<double>());
                    return ret_cam;
                  }))
  .def(py::init([](EigenArray &center, PyRotation &rotation)
                  {
                    Eigen::Vector3d vec(center.getMatrixD().data());
                    kwiver::vital::simple_camera_perspective ret_cam(vec, rotation.getRotD());
                    return ret_cam;
                  }))
  .def(py::init([](EigenArray &center, PyRotation &rotation, py::object &int_obj)
                  {
                    Eigen::Vector3d vec(center.getMatrixD().data());
                    kwiver::vital::simple_camera_intrinsics c_int = int_obj.cast<kwiver::vital::simple_camera_intrinsics>();
                    kwiver::vital::simple_camera_perspective ret_cam(vec, rotation.getRotD(), c_int);
                    return ret_cam;
                  }))
  .def("as_matrix", &kwiver::vital::simple_camera_perspective::as_matrix)
  .def("as_string", [](kwiver::vital::simple_camera_perspective &self)
                      {
                        std::ostringstream ss;
                        ss << self;
                        return ss.str();
                      })
  .def_static("from_string", [](std::string str)
                      {
                        kwiver::vital::simple_camera_perspective self;
                        std::istringstream ss(str);
                        ss >> self;
                        return self;
                      })
  .def("clone_look_at", &kwiver::vital::simple_camera_perspective::clone_look_at,
    py::arg("stare_point"), py::arg("up_direction")=Eigen::Matrix<double,3,1>::UnitZ())
  .def("project", &kwiver::vital::simple_camera_perspective::project,
    py::arg("pt"))
  .def("depth", &kwiver::vital::simple_camera_perspective::depth)
  .def("write_krtd_file", [](kwiver::vital::simple_camera_perspective &self, std::string path)
                                {
                                  kwiver::vital::write_krtd_file(self, path);
                                })
  .def_static("from_krtd_file", [](std::string path)
                                {
                                  kwiver::vital::camera_perspective_sptr c(kwiver::vital::read_krtd_file(path));
                                  return c;
                                })
  .def_property("center", &kwiver::vital::simple_camera_perspective::center,
                          &kwiver::vital::simple_camera_perspective::set_center)
  .def_property("covariance", [](kwiver::vital::simple_camera_perspective &self)
                                {
                                  return PyCovariance3d(self.get_center_covar().matrix());
                                },
                              [](kwiver::vital::simple_camera_perspective &self, PyCovariance3d val)
                                {
                                  self.set_center_covar(val.get_covar());
                                })
  .def_property("translation", &kwiver::vital::simple_camera_perspective::translation,
                               &kwiver::vital::simple_camera_perspective::set_translation)
  .def_property("rotation", [](kwiver::vital::simple_camera_perspective &self)
                                {
                                  PyRotation rot;
                                  rot.setType('d');
                                  rot.setRotD(self.get_rotation());
                                  return rot;
                                },
                            [](kwiver::vital::simple_camera_perspective &self, PyRotation val)
                                {
                                  self.set_rotation(val.getRotD());
                                })
  .def("__eq__", &camera_eq,
    py::arg("other"))
  .def("__ne__", &camera_ne,
    py::arg("other"))
  ;

}
