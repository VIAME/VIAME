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

#include <vital/types/covariance.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class PyCovarianceBase
{

  public:

    virtual ~PyCovarianceBase() = default;

    static py::object covar_from_scalar(int, char, py::object);
    static py::object covar_from_matrix(int, char, py::object);

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

    py::object to_matrix() {return py::cast<Eigen::Matrix<double, 2, 2>>(covar.matrix());};

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

    py::object to_matrix() {return py::cast<Eigen::Matrix<double, 3, 3>>(covar.matrix());};

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

    py::object to_matrix() {return py::cast<Eigen::Matrix<float, 2, 2>>(covar.matrix());};

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

    py::object to_matrix() {return py::cast<Eigen::Matrix<float, 3, 3>>(covar.matrix());};

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

// Unfortunately, because pybind11 can't deal with templates, we have to enumerate
// all the possibilities. But by handling it at construction time, we don't need
// to worry about this later.
py::object
PyCovarianceBase::
covar_from_scalar(int N, char c_type, py::object init)
{
  py::object retVal;
  if(N == 2 && c_type == 'd')
  {
    if(!init.is(py::none()))
    {
      double mat = init.cast< double >();
      retVal = py::cast<PyCovariance2d>(PyCovariance2d(mat));
    }
    else
    {
      retVal = py::cast<PyCovariance2d>(PyCovariance2d());
    }
  }
  else if(N == 3 && c_type == 'd')
  {
    if(!init.is(py::none()))
    {
      double mat = init.cast< double >();
      retVal = py::cast<PyCovariance3d>(PyCovariance3d(mat));
    }
    else
    {
      retVal = py::cast<PyCovariance3d>(PyCovariance3d());
    }
  }
  else if(N == 2 && c_type == 'f')
  {
    if(!init.is(py::none()))
    {
      float mat = init.cast< float >();
      retVal = py::cast<PyCovariance2f>(PyCovariance2f(mat));
    }
    else
    {
      retVal = py::cast<PyCovariance2f>(PyCovariance2f());
    }
  }
  else if(N == 3 && c_type == 'f')
  {
    if(!init.is(py::none()))
    {
      float mat = init.cast< float >();
      retVal = py::cast<PyCovariance3f>(PyCovariance3f(mat));
    }
    else
    {
      retVal = py::cast<PyCovariance3f>(PyCovariance3f());
    }
  }
  return retVal;
}

py::object
PyCovarianceBase::
covar_from_matrix(int N, char c_type, py::object init)
{
  py::object retVal;
  if(N == 2 && c_type == 'd')
  {
    Eigen::Matrix< double, 2, 2 > mat = init.cast< Eigen::Matrix< double, 2, 2 > >();
    retVal = py::cast<PyCovariance2d>(PyCovariance2d(mat));
  }
  else if(N == 3 && c_type == 'd')
  {
    Eigen::Matrix< double, 3, 3 > mat = init.cast< Eigen::Matrix< double, 3, 3 > >();
    retVal = py::cast<PyCovariance3d>(PyCovariance3d(mat));
  }
  else if(N == 2 && c_type == 'f')
  {
    Eigen::Matrix< float, 2, 2 > mat = init.cast< Eigen::Matrix< float, 2, 2 > >();
    retVal = py::cast<PyCovariance2f>(PyCovariance2f(mat));
  }
  else if(N == 3 && c_type == 'f')
  {
    Eigen::Matrix< float, 3, 3 > mat = init.cast< Eigen::Matrix< float, 3, 3 > >();
    retVal = py::cast<PyCovariance3f>(PyCovariance3f(mat));
  }
  return retVal;
}

PYBIND11_MODULE(_covariance, m)
{
  py::class_<PyCovarianceBase, std::shared_ptr<PyCovarianceBase> >(m, "Covariance")
  .def_static("new_covar", &PyCovarianceBase::covar_from_scalar, // need to use a factory func instead of constructor
       py::arg("N")=2, py::arg("c_type")='d', py::arg("init")=py::none())
  .def_static("from_matrix", &PyCovarianceBase::covar_from_matrix,
       py::arg("N")=2, py::arg("c_type")='d', py::arg("init")=py::none())
  .def("to_matrix", &PyCovarianceBase::to_matrix)
  .def("__setitem__", [](PyCovarianceBase &self, py::tuple idx, py::object value)
                      {
                        self.set_item(idx[0].cast<int>(), idx[1].cast<int>(), value);
                      })
  .def("__getitem__", [](PyCovarianceBase &self, py::tuple idx)
                      {
                        return self.get_item(idx[0].cast<int>(), idx[1].cast<int>());
                      })
   ;

  py::class_<PyCovariance2d, PyCovarianceBase, std::shared_ptr<PyCovariance2d>>(m, "Covariance2d");
  py::class_<PyCovariance3d, PyCovarianceBase, std::shared_ptr<PyCovariance3d>>(m, "Covariance3d");
  py::class_<PyCovariance2f, PyCovarianceBase, std::shared_ptr<PyCovariance2f>>(m, "Covariance2f");
  py::class_<PyCovariance3f, PyCovarianceBase, std::shared_ptr<PyCovariance3f>>(m, "Covariance3f");
}
