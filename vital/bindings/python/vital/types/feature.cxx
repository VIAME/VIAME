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

#include <vital/types/feature.h>

#include "covariance_class.cxx"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py=pybind11;

// TODO
// We don't actually need all this extra crap, I wrote this class
// when trying to figure out how to deal with templates in pybind11
// Use one of the other (better!) bindings as a guide to redo.
// When we redo this, make sure to update feature_track_state to match
class PyFeatureBase
{
  public:

    virtual ~PyFeatureBase() = default;

    virtual char get_type() {return '0';};

    virtual py::object get_loc() {return py::none();};
    virtual void set_loc(py::object) {};

    virtual py::object get_mag() {return py::none();};
    virtual void set_mag(py::object) {};

    virtual py::object get_scale() {return py::none();};
    virtual void set_scale(py::object) {};

    virtual py::object get_angle() {return py::none();};
    virtual void set_angle(py::object) {};

    virtual PyCovarianceBase get_covar() = 0;
    virtual void set_covar(py::object) {};

    virtual kwiver::vital::rgb_color get_color() {return kwiver::vital::rgb_color();};
    virtual void set_color(kwiver::vital::rgb_color) {};

    virtual kwiver::vital::feature_sptr get_feature() = 0;
};

class PyFeatureD
: public PyFeatureBase
{

  kwiver::vital::feature_<double> feature;

  public:

    PyFeatureD(Eigen::Matrix<double,2,1> loc,
               double mag,
               double scale,
               double angle,
               kwiver::vital::rgb_color color)
              {
                feature = kwiver::vital::feature_<double>(loc, mag, scale, angle, color);
              }

    char get_type() { return 'd';};

    py::object get_loc() { return py::cast<Eigen::Matrix<double,2,1>>(feature.get_loc()); };
    void set_loc(py::object loc_obj) { feature.set_loc(loc_obj.cast<Eigen::Matrix<double,2,1>>()); };

    py::object get_mag() { return py::cast<double>(feature.get_magnitude()); };
    void set_mag(py::object mag_obj) { feature.set_magnitude(mag_obj.cast<double>()); };

    py::object get_scale() { return py::cast<double>(feature.get_scale()); };
    void set_scale(py::object scale_obj) { feature.set_scale(scale_obj.cast<double>()); };

    py::object get_angle() { return py::cast<double>(feature.get_angle()); };
    void set_angle(py::object angle_obj) { feature.set_angle(angle_obj.cast<double>()); };

    PyCovarianceBase get_covar()
                     {
                       return PyCovariance2d(feature.get_covar());
                     };
    void set_covar(py::object mat_obj)
                     {
                       PyCovariance2d covar;
                       try // we need to check to see if it's a matrix or a covariance object
                       {
                         Eigen::Matrix<double, 2, 2> mat = mat_obj.cast<Eigen::Matrix<double,2,2>>();
                         covar = PyCovariance2d(mat);
                       }
                       catch (...) // if matrix doesn't work, try covar object
                       {
                         covar = mat_obj.cast<PyCovariance2d>();
                       }

                       feature.set_covar(covar.get_covar());
                     };

    kwiver::vital::rgb_color get_color() { return feature.get_color(); };
    void set_color(kwiver::vital::rgb_color color) { feature.set_color(color); };

    kwiver::vital::feature_sptr get_feature() {return kwiver::vital::feature_sptr(&feature);};
};

class PyFeatureF
: public PyFeatureBase
{

  kwiver::vital::feature_<float> feature;

  public:

    PyFeatureF(Eigen::Matrix<float,2,1> loc,
               float mag,
               float scale,
               float angle,
               kwiver::vital::rgb_color color)
              {
                feature = kwiver::vital::feature_<float>(loc, mag, scale, angle, color);
              }

    char get_type() { return 'f';};

    py::object get_loc() { return py::cast<Eigen::Matrix<float,2,1>>(feature.get_loc()); };
    void set_loc(py::object loc_obj) { feature.set_loc(loc_obj.cast<Eigen::Matrix<float,2,1>>()); };

    py::object get_mag() { return py::cast<float>(feature.get_magnitude()); };
    void set_mag(py::object mag_obj) { feature.set_magnitude(mag_obj.cast<float>()); };

    py::object get_scale() { return py::cast<float>(feature.get_scale()); };
    void set_scale(py::object scale_obj) { feature.set_scale(scale_obj.cast<float>()); };

    py::object get_angle() { return py::cast<float>(feature.get_angle()); };
    void set_angle(py::object angle_obj) { feature.set_angle(angle_obj.cast<float>()); };

    PyCovarianceBase get_covar()
                     {
                       return PyCovariance2f(feature.get_covar());
                     };
    void set_covar(py::object mat_obj)
                     {
                       PyCovariance2f covar;
                       try // we need to check to see if it's a matrix or a covariance object
                       {
                         Eigen::Matrix<float, 2, 2> mat = mat_obj.cast<Eigen::Matrix<float,2,2>>();
                         covar = PyCovariance2f(mat);
                       }
                       catch (...) // if matrix doesn't work, try covar object
                       {
                         covar = mat_obj.cast<PyCovariance2f>();
                       }

                       feature.set_covar(covar.get_covar());
                     };


    kwiver::vital::rgb_color get_color() { return feature.get_color(); };
    void set_color(kwiver::vital::rgb_color color) { feature.set_color(color); };

    kwiver::vital::feature_sptr get_feature() {return kwiver::vital::feature_sptr(&feature);};
};

std::shared_ptr<PyFeatureBase>
new_feature(py::object loc_obj,
            py::object mag_obj,
            py::object scale_obj,
            py::object angle_obj,
            kwiver::vital::rgb_color color,
            char ctype)
{
  std::shared_ptr<PyFeatureBase> retVal;
  if(ctype == 'd')
  {
    Eigen::Matrix<double,2,1> loc = loc_obj.is(py::none()) ? Eigen::Matrix<double,2,1>() : loc_obj.cast<Eigen::Matrix<double,2,1>>();
    double mag = mag_obj.cast<double>();
    double scale = scale_obj.cast<double>();
    double angle = angle_obj.cast<double>();
    retVal = std::shared_ptr<PyFeatureBase>(new PyFeatureD(loc, mag, scale, angle, color));
  }
  else if(ctype == 'f')
  {
    Eigen::Matrix<float,2,1> loc = loc_obj.is(py::none()) ? Eigen::Matrix<float,2,1>() : loc_obj.cast<Eigen::Matrix<float,2,1>>();
    float mag = mag_obj.cast<float>();
    float scale = scale_obj.cast<float>();
    float angle = angle_obj.cast<float>();
    retVal = std::shared_ptr<PyFeatureBase>(new PyFeatureF(loc, mag, scale, angle, color));
  }

  return retVal;
}

PYBIND11_MODULE(feature, m)
{
  py::class_<PyFeatureBase, std::shared_ptr<PyFeatureBase>>(m, "Feature")
  .def(py::init(&new_feature),
    py::arg("loc")=py::none(), py::arg("mag")=0., py::arg("scale")=1.,
    py::arg("angle")=0., py::arg("rgb_color")=kwiver::vital::rgb_color(), py::arg("ctype")='d')
  .def_property_readonly("type_name", &PyFeatureBase::get_type)
  .def_property("location", &PyFeatureBase::get_loc, &PyFeatureBase::set_loc)
  .def_property("magnitude", &PyFeatureBase::get_mag, &PyFeatureBase::set_mag)
  .def_property("scale", &PyFeatureBase::get_scale, &PyFeatureBase::set_scale)
  .def_property("angle", &PyFeatureBase::get_angle, &PyFeatureBase::set_angle)
  .def_property("covariance", &PyFeatureBase::get_covar, &PyFeatureBase::set_covar)
  .def_property("color", &PyFeatureBase::get_color, &PyFeatureBase::set_color)
  ;
}
