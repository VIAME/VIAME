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
#ifndef PY_COVARIANCE_CLASS_H
#define PY_COVARIANCE_CLASS_H


#include <vital/types/covariance.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyCovarianceBase
{

  public:

    virtual ~PyCovarianceBase() = default;

    static std::shared_ptr<PyCovarianceBase> covar_from_scalar(int, char, py::object);
    static std::shared_ptr<PyCovarianceBase> covar_from_matrix(int, char, py::object);

    virtual py::object to_matrix() {return py::none();};

    virtual void set_item(int, int, py::object) {};
    virtual py::object get_item(int, int) {return py::none();};
};

class PyCovariance2d
: public PyCovarianceBase
{

  kwiver::vital::covariance_<2, double> covar;

  public:

    PyCovariance2d() {covar = kwiver::vital::covariance_<2, double>();};
    PyCovariance2d(double mat)
                  {covar = kwiver::vital::covariance_<2, double>(mat);};
    PyCovariance2d(Eigen::Matrix<double, 2, 2> mat)
                  {covar = kwiver::vital::covariance_<2, double>(mat);};
    PyCovariance2d(kwiver::vital::covariance_<2, double> mat)
                  {covar = mat;};

    py::object to_matrix() {return py::cast<Eigen::Matrix<double, 2, 2>>(covar.matrix());};
    kwiver::vital::covariance_<2, double> get_covar() {return covar;};

    void set_item(int i, int j, py::object value)
                                    {
                                      if(i < 0 || i > 1 || j < 0 || j > 1)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      covar(i, j) = value.cast<double>();
                                    };
    py::object get_item(int i, int j)
                                    {
                                      if(i < 0 || i > 1 || j < 0 || j > 1)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      return py::cast<double>(covar(i, j));
                                    };

};

class PyCovariance3d
: public PyCovarianceBase
{

  kwiver::vital::covariance_<3, double> covar;

  public:

    PyCovariance3d() {covar = kwiver::vital::covariance_<3, double>();};
    PyCovariance3d(double mat)
                  {covar = kwiver::vital::covariance_<3, double>(mat);};
    PyCovariance3d(Eigen::Matrix<double, 3, 3> mat)
                  {covar = kwiver::vital::covariance_<3, double>(mat);};
    PyCovariance3d(kwiver::vital::covariance_<3, double> mat)
                  {covar = mat;};

    py::object to_matrix() {return py::cast<Eigen::Matrix<double, 3, 3>>(covar.matrix());};
    kwiver::vital::covariance_<3, double> get_covar() {return covar;};

    void set_item(int i, int j, py::object value)
                                    {
                                      if(i < 0 || i > 2 || j < 0 || j > 2)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      covar(i, j) = value.cast<double>();
                                    };
    py::object get_item(int i, int j)
                                    {
                                      if(i < 0 || i > 2 || j < 0 || j > 2)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      return py::cast<double>(covar(i, j));
                                    };
};

class PyCovariance2f
: public PyCovarianceBase
{

  kwiver::vital::covariance_<2, float> covar;

  public:

    PyCovariance2f() {covar = kwiver::vital::covariance_<2, float>();};
    PyCovariance2f(float mat)
                  {covar = kwiver::vital::covariance_<2, float>(mat);};
    PyCovariance2f(Eigen::Matrix<float, 2, 2> mat)
                  {covar = kwiver::vital::covariance_<2, float>(mat);};
    PyCovariance2f(kwiver::vital::covariance_<2, float> mat)
                  {covar = mat;};

    py::object to_matrix() {return py::cast<Eigen::Matrix<float, 2, 2>>(covar.matrix());};
    kwiver::vital::covariance_<2, float> get_covar() {return covar;};

    void set_item(int i, int j, py::object value)
                                    {
                                      if(i < 0 || i > 1 || j < 0 || j > 1)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      covar(i, j) = value.cast<float>();
                                    };
    py::object get_item(int i, int j)
                                    {
                                      if(i < 0 || i > 1 || j < 0 || j > 1)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      return py::cast<float>(covar(i, j));
                                    };
};

class PyCovariance3f
: public PyCovarianceBase
{

  kwiver::vital::covariance_<3, float> covar;

  public:

    PyCovariance3f() {covar = kwiver::vital::covariance_<3, float>();};
    PyCovariance3f(float mat)
                  {covar = kwiver::vital::covariance_<3, float>(mat);};
    PyCovariance3f(Eigen::Matrix<float, 3, 3> mat)
                  {covar = kwiver::vital::covariance_<3, float>(mat);};
    PyCovariance3f(kwiver::vital::covariance_<3, float> mat)
                  {covar = mat;};

    py::object to_matrix() {return py::cast<Eigen::Matrix<float, 3, 3>>(covar.matrix());};
    kwiver::vital::covariance_<3, float> get_covar() {return covar;};

    void set_item(int i, int j, py::object value)
                                    {
                                      if(i < 0 || i > 2 || j < 0 || j > 2)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      covar(i, j) = value.cast<float>();
                                    };
    py::object get_item(int i, int j)
                                    {
                                      if(i < 0 || i > 2 || j < 0 || j > 2)
                                      {
                                        throw py::index_error("Index out of range!");
                                      }
                                      return py::cast<float>(covar(i, j));
                                    };
};

#endif
