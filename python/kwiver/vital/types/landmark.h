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
#ifndef PY_LANDMARK_H
#define PY_LANDMARK_H

#include <vital/types/landmark.h>

#include <python/kwiver/vital/types/covariance_class.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py=pybind11;

void landmark(py::module &m);

class PyLandmarkBase
{
  public:

    virtual ~PyLandmarkBase() = default;

    virtual char get_type() { return '0';};

    virtual py::object get_loc() { return py::none();};
    virtual void set_loc(py::object) { };

    virtual py::object get_scale() { return py::none();};
    virtual void set_scale(py::object) { };

    virtual py::object get_normal() { return py::none();};
    virtual void set_normal(py::object) { };

    virtual py::object get_covariance() { return py::none();};
    virtual void set_covariance(py::object) { };

    virtual kwiver::vital::rgb_color get_color() { return kwiver::vital::rgb_color();};
    virtual void set_color(kwiver::vital::rgb_color) { };

    virtual unsigned get_observations() { return 0;};
    virtual void set_observations(unsigned) { };
};

class PyLandmarkd
: public PyLandmarkBase
{

  kwiver::vital::landmark_d landmark;

  public:

    PyLandmarkd(Eigen::Matrix<double, 3, 1> loc, double scale)
               {
                 landmark = kwiver::vital::landmark_<double>(loc, scale);
               };

    char get_type() { return 'd';};
    py::object get_loc()
            {
              return py::cast<Eigen::Matrix<double,3,1>>(landmark.get_loc());
            };
    void set_loc(py::object loc_obj)
            {
              auto loc = loc_obj.cast<Eigen::Matrix<double,3,1>>();
              landmark.set_loc(loc);
            };

    py::object get_scale()
            {
              return py::cast<double>(landmark.get_scale());
            };
    void set_scale(py::object scale_obj)
            {
              double scale = scale_obj.cast<double>();
              landmark.set_scale(scale);
            };

    py::object get_normal()
            {
              return py::cast<Eigen::Matrix<double,3,1>>(landmark.get_normal());
            };
    void set_normal(py::object norm_obj)
            {
              Eigen::Matrix<double,3,1> norm = norm_obj.cast<Eigen::Matrix<double,3,1>>();
              landmark.set_normal(norm);
            };

    py::object get_covariance()
            {
              return py::cast<PyCovariance3d>(PyCovariance3d(landmark.get_covar()));
            };
    void set_covariance(py::object covar_obj)
            {
              PyCovariance3d covar = covar_obj.cast<PyCovariance3d>();
              landmark.set_covar(covar.get_covar());
            };

    kwiver::vital::rgb_color get_color() { return landmark.get_color(); };
    void set_color(kwiver::vital::rgb_color color) { landmark.set_color(color); };

    unsigned get_observations() { return landmark.get_observations(); };
    void set_observations(unsigned obs) { landmark.set_observations(obs); };
};

class PyLandmarkf
: public PyLandmarkBase
{

  kwiver::vital::landmark_f landmark;

  public:

    PyLandmarkf(Eigen::Matrix<float, 3, 1> loc, float scale)
               {
                 landmark = kwiver::vital::landmark_<float>(loc, scale);
               };

    char get_type() { return 'f';};
    py::object get_loc()
            {
              return py::cast<Eigen::Matrix<float,3,1>>(landmark.get_loc());
            };
    void set_loc(py::object loc_obj)
            {
              auto loc = loc_obj.cast<Eigen::Matrix<float,3,1>>();
              landmark.set_loc(loc);
            };

    py::object get_scale()
            {
              return py::cast<float>(landmark.get_scale());
            };
    void set_scale(py::object scale_obj)
            {
              float scale = scale_obj.cast<float>();
              landmark.set_scale(scale);
            };

    py::object get_normal()
            {
              return py::cast<Eigen::Matrix<float,3,1>>(landmark.get_normal());
            };
    void set_normal(py::object norm_obj)
            {
              Eigen::Matrix<float,3,1> norm = norm_obj.cast<Eigen::Matrix<float,3,1>>();
              landmark.set_normal(norm);
            };

    py::object get_covariance()
            {
              return py::cast<PyCovariance3f>(PyCovariance3f(landmark.get_covar()));
            };
    void set_covariance(py::object covar_obj)
            {
              PyCovariance3f covar = covar_obj.cast<PyCovariance3f>();
              landmark.set_covar(covar.get_covar());
            };

    kwiver::vital::rgb_color get_color() { return landmark.get_color(); };
    void set_color(kwiver::vital::rgb_color color) { landmark.set_color(color); };

    unsigned get_observations() { return landmark.get_observations(); };
    void set_observations(unsigned obs) { landmark.set_observations(obs); };
};

std::shared_ptr<PyLandmarkBase>
new_landmark(py::object loc_obj, py::object scale_obj, char ctype);

#endif
