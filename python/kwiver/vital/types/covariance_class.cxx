// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/covariance.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {

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

// Unfortunately, because pybind11 can't deal with templates, we have to enumerate
// all the possibilities. But by handling it at construction time, we don't need
// to worry about this later.
std::shared_ptr<PyCovarianceBase>
PyCovarianceBase::
covar_from_scalar(int N, char c_type, py::object init)
{
  std::shared_ptr<PyCovarianceBase> retVal;
  if(N == 2 && c_type == 'd')
  {
    if(!init.is(py::none()))
    {
      double mat = init.cast< double >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2d(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2d());
    }
  }
  else if(N == 3 && c_type == 'd')
  {
    if(!init.is(py::none()))
    {
      double mat = init.cast< double >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3d(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3d());
    }
  }
  else if(N == 2 && c_type == 'f')
  {
    if(!init.is(py::none()))
    {
      float mat = init.cast< float >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2f(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2f());
    }
  }
  else if(N == 3 && c_type == 'f')
  {
    if(!init.is(py::none()))
    {
      float mat = init.cast< float >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3f(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3f());
    }
  }
  return retVal;
}

std::shared_ptr<PyCovarianceBase>
PyCovarianceBase::
covar_from_matrix(int N, char c_type, py::object init)
{
  std::shared_ptr<PyCovarianceBase> retVal;
  if(N == 2 && c_type == 'd')
  {
    Eigen::Matrix< double, 2, 2 > mat = init.cast< Eigen::Matrix< double, 2, 2 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2d(mat));
  }
  else if(N == 3 && c_type == 'd')
  {
    Eigen::Matrix< double, 3, 3 > mat = init.cast< Eigen::Matrix< double, 3, 3 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3d(mat));
  }
  else if(N == 2 && c_type == 'f')
  {
    Eigen::Matrix< float, 2, 2 > mat = init.cast< Eigen::Matrix< float, 2, 2 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2f(mat));
  }
  else if(N == 3 && c_type == 'f')
  {
    Eigen::Matrix< float, 3, 3 > mat = init.cast< Eigen::Matrix< float, 3, 3 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3f(mat));
  }
  return retVal;
}
}
}
}
