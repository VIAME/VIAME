// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera_intrinsics.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

typedef kwiver::vital::simple_camera_intrinsics cam_int;

PYBIND11_MODULE(camera_intrinsics, m)
{

  py::class_<cam_int, std::shared_ptr<cam_int> >(m, "CameraIntrinsics")
  .def(py::init<const double,
                const kwiver::vital::vector_2d&,
                const double,
                const double,
                const Eigen::VectorXd>(),
       py::arg("focal_length")=1.0,
       py::arg("principal_point")=kwiver::vital::vector_2d(0.0,0.0),
       py::arg("aspect_ratio")=1.0,
       py::arg("skew")=0.0,
       py::arg("dist_coeffs")=Eigen::VectorXd::Zero(1))
  .def("as_matrix", &cam_int::as_matrix)
  .def("__eq__", [](const std::shared_ptr<cam_int> self,
                    const std::shared_ptr<cam_int> other)
                   {
                     return(self->get_aspect_ratio() == other->get_aspect_ratio() &&
                            self->get_dist_coeffs() == other->get_dist_coeffs() &&
                            self->get_focal_length() == other->get_focal_length() &&
                            self->get_principal_point() == other->get_principal_point() &&
                            self->get_skew() == other->get_skew());
                   })
  .def("__ne__", [](const std::shared_ptr<cam_int> self,
                    const std::shared_ptr<cam_int> other)
                   {
                     return(self->get_aspect_ratio() != other->get_aspect_ratio() ||
                            self->get_dist_coeffs() != other->get_dist_coeffs() ||
                            self->get_focal_length() != other->get_focal_length() ||
                            self->get_principal_point() != other->get_principal_point() ||
                            self->get_skew() != other->get_skew());
                   })
  .def_property("aspect_ratio", &cam_int::get_aspect_ratio, &cam_int::set_aspect_ratio)
  .def_property("dist_coeffs", &cam_int::get_dist_coeffs, &cam_int::set_dist_coeffs)
  .def_property("focal_length", &cam_int::get_focal_length, &cam_int::set_focal_length)
  .def_property("principal_point", &cam_int::get_principal_point, &cam_int::set_principal_point)
  .def_property("skew", &cam_int::get_skew, &cam_int::set_skew)
  ;
}
