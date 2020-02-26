// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/landmark.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


std::shared_ptr<PyLandmarkBase>
new_landmark(py::object loc_obj, py::object scale_obj, char ctype)
{
  std::shared_ptr<PyLandmarkBase> retVal;
  if(ctype == 'd')
  {
    // Get our arguments, taking care of default cases
    Eigen::Matrix<double, 3, 1> loc = Eigen::Matrix<double, 3, 1>();
    loc << 0,0,0;
    if(!loc_obj.is(py::none()))
    {
      loc = loc_obj.cast<Eigen::Matrix<double, 3, 1>>();
    }

    double scale = 1.;
    if(!scale_obj.is(py::none()))
    {
      scale = scale_obj.cast<double>();
    }

    retVal = std::shared_ptr<PyLandmarkBase>(new PyLandmarkd(loc, scale));
  }
  else if(ctype == 'f')
  {
    // Get our arguments, taking care of default cases
    Eigen::Matrix<float, 3, 1> loc = Eigen::Matrix<float, 3, 1>();
    loc << 0,0,0;
    if(!loc_obj.is(py::none()))
    {
      loc = loc_obj.cast<Eigen::Matrix<float, 3, 1>>();
    }

    float scale = 1.;
    if(!scale_obj.is(py::none()))
    {
      scale = scale_obj.cast<float>();
    }

    retVal = std::shared_ptr<PyLandmarkBase>(new PyLandmarkf(loc, scale));
  }

  return retVal;
}

void landmark(py::module &m)
{

  py::class_<PyLandmarkBase, std::shared_ptr<PyLandmarkBase>>(m, "Landmark")
  .def(py::init(&new_landmark),
    py::arg("loc")=py::none(), py::arg("scale")=py::none(), py::arg("ctype")='d')
  .def_property_readonly("type_name", &PyLandmarkBase::get_type)
  .def_property("loc", &PyLandmarkBase::get_loc, &PyLandmarkBase::set_loc)
  .def_property("scale", &PyLandmarkBase::get_scale, &PyLandmarkBase::set_scale)
  .def_property("normal", &PyLandmarkBase::get_normal, &PyLandmarkBase::set_normal)
  .def_property("covariance", &PyLandmarkBase::get_covariance, &PyLandmarkBase::set_covariance)
  .def_property("color", &PyLandmarkBase::get_color, &PyLandmarkBase::set_color)
  .def_property("observations", &PyLandmarkBase::get_observations, &PyLandmarkBase::set_observations)
  .def_property("cos_obs_angle", &PyLandmarkBase::get_cos_obs_angle, &PyLandmarkBase::set_cos_obs_angle)
  ;

}
