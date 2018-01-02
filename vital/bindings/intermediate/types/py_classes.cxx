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

#include <Eigen/Core>

#include <vital/types/covariance.h>
#include <vital/types/rotation.h>
#include <vital/types/descriptor.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class EigenArray
{
  // We're assuming always dynamic, to make things simpler for first pass
  // TODO We can edit this later to use subclasses instead of two parallel
  Eigen::MatrixXd double_mat;
  Eigen::MatrixXf float_mat;
  char type;

  public:

    EigenArray(unsigned int, unsigned int, bool, bool, char);

    void fromVectorF(std::vector< std::vector<float> >);
    void fromVectorD(std::vector< std::vector<double> >);
    static EigenArray fromArray(py::object, char);

    void setType(char ctype) { this->type = ctype; };
    char getType() { return type; };

    py::object getMatrix();
    Eigen::MatrixXd getMatrixD() { return double_mat; };
    Eigen::MatrixXf getMatrixF() { return float_mat; };
    void setMatrixF();
    void setMatrixD();

};

EigenArray::
EigenArray(unsigned int rows = 2,
             unsigned int cols = 1,
             bool dynamic_rows = false, // we're ignoring these, but keeping them in for API reasons
             bool dynamic_cols = false,
             char ctype = 'd')
{
  type = ctype;
  if(type == 'd') double_mat = Eigen::MatrixXd(rows, cols);
  else if(type == 'f') float_mat = Eigen::MatrixXf(rows, cols);
  else throw std::invalid_argument("Invalid matrix type. Must be 'd' or 'f'");
}

py::object
EigenArray::
getMatrix()
{
  if(this->getType() == 'd') return py::cast(&double_mat);
  else if(this->getType() == 'f') return py::cast(&float_mat);
  return py::none();
}

void
EigenArray::
fromVectorF(std::vector< std::vector<float> > data_vec)
{
  unsigned int rows = data_vec.size();
  unsigned int cols = data_vec[0].size();
  this->float_mat = Eigen::MatrixXf(rows, cols);
  for(unsigned int i = 0; i < rows; i++)
  {
    if (data_vec[i].size() != cols)
    {
       throw std::invalid_argument("Input is not an mxn matrix!");
    }
    for(unsigned int j = 0; j < cols; j++)
    {
      float_mat(i,j) = data_vec[i][j];
    }
  }
}

void
EigenArray::
fromVectorD(std::vector< std::vector<double> > data_vec)
{
  unsigned int rows = data_vec.size();
  unsigned int cols = data_vec[0].size();
  this->double_mat = Eigen::MatrixXd(rows, cols);
  for(unsigned int i = 0; i < rows; i++)
  {
    if (data_vec[i].size() != cols)
    {
       throw std::invalid_argument("Input is not an mxn matrix!");
    }
    for(unsigned int j = 0; j < cols; j++)
    {
      double_mat(i,j) = data_vec[i][j];
    }
  }
}

EigenArray
EigenArray::
fromArray(py::object data, char ctype = 'd')
{
  EigenArray retMat;

  retMat.setType(ctype);

  if(ctype == 'd')
  {
    std::vector< std::vector<double> > data_vec = data.cast<std::vector< std::vector<double> > >();
    retMat.fromVectorD(data_vec);
  }
  else if(ctype == 'f')
  {
    std::vector< std::vector<float> > data_vec = data.cast<std::vector< std::vector<float> > >();
    retMat.fromVectorF(data_vec);
  }
  else
  {
    throw std::invalid_argument("Invalid matrix type. Must be 'd' or 'f'");
  }

  return retMat;
}

class PyRotation
{
  // This is done to get around pybind11's dislike of templated classes
  // TODO We can edit this later to use subclasses instead of two parallel
  kwiver::vital::rotation_<double> double_rot;
  kwiver::vital::rotation_<float> float_rot;
  char type;

  public:

    PyRotation(char);

    void fromQuatF(const Eigen::Quaternion< float >);
    void fromQuatD(const Eigen::Quaternion< double >);
    static PyRotation from_quaternion(py::object, char);
    py::object quaternion();

    static PyRotation from_axis_angle(py::object, py::object, char);
    py::object angle();
    py::object axis();

    static PyRotation from_matrix(py::object, char);
    py::object matrix();

    static PyRotation from_rodrigues(py::object, char);
    py::object rodrigues();

    static PyRotation from_ypr(py::object, py::object, py::object, char);
    py::list get_yaw_pitch_roll();

    static PyRotation interpolate(PyRotation, PyRotation, py::object);
    static std::vector<PyRotation> interpolated_rotations(PyRotation, PyRotation, size_t);

    void setType(char ctype) { this->type = ctype; };
    char getType() { return type; };

    void setRotD(kwiver::vital::rotation_<double> new_rot) { double_rot = new_rot;};
    void setRotF(kwiver::vital::rotation_<float> new_rot) { float_rot = new_rot;};
    kwiver::vital::rotation_<double> getRotD() { return double_rot; };
    kwiver::vital::rotation_<float> getRotF() { return float_rot; };

    void convert_to_d();
    void convert_to_f();

    py::object angle_from(const std::shared_ptr<PyRotation>);
    PyRotation compose(const std::shared_ptr<PyRotation>);
    PyRotation inverse();
    py::object rotate_vector(py::object);
};

PyRotation::
PyRotation(char ctype = 'd')
{
  type = ctype;
  if(type == 'd') double_rot = kwiver::vital::rotation_<double>();
  else if(type == 'f') float_rot = kwiver::vital::rotation_<float>();
  else throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
}

void
PyRotation::
convert_to_d()
{
  if(type == 'd') // don't worry about converting if we're already the right type
  {
    return;
  }

  type = 'd';

  Eigen::Quaternion<float> float_quat = float_rot.quaternion();
  Eigen::Quaternion<double> double_quat(float_quat);
  double_rot = kwiver::vital::rotation_<double>(double_quat);
}

void
PyRotation::
convert_to_f()
{
  if(type == 'f') // don't worry about converting if we're already the right type
  {
    return;
  }

  type = 'f';

  Eigen::Quaternion<double> double_quat = double_rot.quaternion();
  Eigen::Quaternion<float> float_quat(double_quat);
  float_rot = kwiver::vital::rotation_<float>(float_quat);
}

void
PyRotation::
fromQuatF(const Eigen::Quaternion< float > quat)
{
  float_rot = kwiver::vital::rotation_<float>(quat);
}

void
PyRotation::
fromQuatD(const Eigen::Quaternion< double > quat)
{
  double_rot = kwiver::vital::rotation_<double>(quat);
}

PyRotation
PyRotation::
from_axis_angle(py::object axis_obj, py::object angle_obj, char ctype = 'd')
{
  PyRotation retRot;

  retRot.setType(ctype);

  if(ctype == 'd')
  {
    Eigen::Matrix< double, 3, 1 > axis = axis_obj.cast<Eigen::Matrix< double, 3, 1 > >();
    double angle = angle_obj.cast<double>();
    retRot.setRotD(kwiver::vital::rotation_<double>(angle, axis));
  }
  else if(ctype == 'f')
  {
    Eigen::Matrix< float, 3, 1 > axis = axis_obj.cast<Eigen::Matrix< float, 3, 1 > >();
    float angle = angle_obj.cast<float>();
    retRot.setRotF(kwiver::vital::rotation_<float>(angle, axis));
  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retRot;
}

PyRotation
PyRotation::
from_matrix(py::object mat_obj, char ctype='d')
{
  PyRotation retRot;

  retRot.setType(ctype);

  if(ctype == 'd')
  {
    Eigen::Matrix< double, 3, 3 > matrix = mat_obj.cast<Eigen::Matrix< double, 3, 3 > >();
    kwiver::vital::rotation_<double> rot(matrix);
    retRot.setRotD(rot);
  }
  else if(ctype == 'f')
  {
    Eigen::Matrix< float, 3, 3 > matrix = mat_obj.cast<Eigen::Matrix< float, 3, 3 > >();
    kwiver::vital::rotation_<float> rot(matrix);
    retRot.setRotF(rot);
  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retRot;
}

PyRotation
PyRotation::
from_quaternion(py::object data, char ctype = 'd')
{
  PyRotation retRot;

  retRot.setType(ctype);

  if(ctype == 'd')
  {
    std::vector< double > data_quat = data.cast<std::vector< double > >();
    if(data_quat.size() != 4)
    {
      throw std::invalid_argument("Quaternion argument must be an array with four elements.");
    }
    retRot.fromQuatD(Eigen::Quaternion< double > (data_quat.data()));
  }
  else if(ctype == 'f')
  {
    std::vector< float > data_quat = data.cast<std::vector< float > >();
    if(data_quat.size() != 4)
    {
      throw std::invalid_argument("Quaternion argument must be an array with four elements.");
    }
    retRot.fromQuatF(Eigen::Quaternion< float > (data_quat.data()));

  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retRot;
}

PyRotation
PyRotation::
from_rodrigues(py::object rod_obj, char ctype='d')
{
  PyRotation retRot;

  retRot.setType(ctype);

  if(ctype == 'd')
  {
    Eigen::Matrix< double, 3, 1 > rodrigues = rod_obj.cast<Eigen::Matrix< double, 3, 1 > >();
    kwiver::vital::rotation_<double> rot(rodrigues);
    retRot.setRotD(rot);    
  }
  else if(ctype == 'f')
  {
    Eigen::Matrix< float, 3, 1 > rodrigues = rod_obj.cast<Eigen::Matrix< float, 3, 1 > >();
    kwiver::vital::rotation_<float> rot(rodrigues);
    retRot.setRotF(rot);    
  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retRot;
}

PyRotation
PyRotation::
from_ypr(py::object yaw, py::object pitch, py::object roll, char ctype='d')
{
  PyRotation retRot;

  retRot.setType(ctype);

  if(ctype == 'd')
  {
    double y = yaw.cast<double>();
    double p = pitch.cast<double>();
    double r = roll.cast<double>();
    kwiver::vital::rotation_<double> rot(y,p,r);
    retRot.setRotD(rot);    
  }
  else if(ctype == 'f')
  {
    float y = yaw.cast<float>();
    float p = pitch.cast<float>();
    float r = roll.cast<float>();
    kwiver::vital::rotation_<float> rot(y,p,r);
    retRot.setRotF(rot);    
  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retRot;
}

PyRotation
PyRotation::
interpolate(PyRotation A, PyRotation B, py::object f_obj)
{
  PyRotation retRot;

  retRot.setType(A.getType());

  if(A.getType() == 'd')
  {
    if(B.getType() == 'f')
    {
      B.convert_to_d();
      B.setType('f');
    }

    double f = f_obj.cast<double>();
    retRot.setRotD(kwiver::vital::interpolate_rotation<double>(A.getRotD(), B.getRotD(), f));
  }
  else if(A.getType() == 'f')
  {
    if(B.getType() == 'd')
    {
      B.convert_to_f();
      B.setType('d');
    }

    float f = f_obj.cast<float>();
    retRot.setRotF(kwiver::vital::interpolate_rotation<float>(A.getRotF(), B.getRotF(), f));
  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retRot;
}

std::vector<PyRotation>
PyRotation::
interpolated_rotations(PyRotation A, PyRotation B, size_t n)
{
  std::vector<PyRotation> retList;

  if(A.getType() == 'd')
  {
    if(B.getType() == 'f')
    {
      B.convert_to_d();
      B.setType('f');
    }

    std::vector<kwiver::vital::rotation_<double>> rots;
    kwiver::vital::interpolated_rotations<double>(A.getRotD(), B.getRotD(), n, rots);
    for(auto rot : rots)
    {
      PyRotation tmp('d');
      tmp.setRotD(rot);
      retList.push_back(tmp);
    }
  }
  else if(A.getType() == 'f')
  {
    if(B.getType() == 'd')
    {
      B.convert_to_f();
      B.setType('d');
    }

    std::vector<kwiver::vital::rotation_<float>> rots;
    kwiver::vital::interpolated_rotations<float>(A.getRotF(), B.getRotF(), n, rots);
    for(auto rot : rots)
    {
      PyRotation tmp('f');
      tmp.setRotF(rot);
      retList.push_back(tmp);
    }
  }
  else
  {
    throw std::invalid_argument("Invalid rotation type. Must be 'd' or 'f'");
  }

  return retList;
}

py::object
PyRotation::
angle()
{
  py::object retVal;

  if(type == 'd')
  {
    retVal = py::cast<double>(double_rot.angle());
  }
  else if(type == 'f')
  {
    retVal = py::cast<float>(float_rot.angle());
  }
  return retVal;
}

py::object
PyRotation::
angle_from(const std::shared_ptr<PyRotation> other)
{
  return this->inverse().compose(other).angle();
}

py::object
PyRotation::
axis()
{
  py::object axis;
  if(type == 'f')
  {
    axis = py::cast<Eigen::Matrix<float,3,1>>(float_rot.axis());
  }
  else if(type == 'd')
  {
    axis = py::cast<Eigen::Matrix<double,3,1>>(double_rot.axis());
  }

  return axis;
}

PyRotation
PyRotation::
compose(const std::shared_ptr<PyRotation> other)
{
  PyRotation retRot;
  if(type == 'd')
  {
    if(other->getType() == 'f') // the types mismatch
    {
      other->convert_to_d();
      other->setType('f'); // we're just populating the other rotation, not fully switching
    }
    retRot.setType('d');
    retRot.setRotD(this->getRotD() * other->getRotD());
  }
  else if(type == 'f')
  {
    if(other->getType() == 'd') // the types mismatch
    {
      other->convert_to_f();
      other->setType('d'); // we're just populating the other rotation, not fully switching
    }
    retRot.setType('f');
    retRot.setRotF(this->getRotF() * other->getRotF());
  }

  return retRot;
}

PyRotation
PyRotation::
inverse()
{
  PyRotation retRot;
  if(type == 'd')
  {
    retRot.setType('d');
    retRot.setRotD(this->getRotD().inverse());
  }
  else if(type == 'f')
  {
    retRot.setType('f');
    retRot.setRotF(this->getRotF().inverse());
  }
  return retRot;
}

py::object
PyRotation::
matrix()
{
  py::object mat;
  if(type == 'f')
  {
    mat = py::cast<Eigen::Matrix<float,3,3>>(float_rot.matrix());
  }
  else if(type == 'd')
  {
    mat = py::cast<Eigen::Matrix<double,3,3>>(double_rot.matrix());
  }

  return mat;
}

py::object
PyRotation::
quaternion()
{
  py::object ret_obj;
  if(type == 'd')
  {
    std::vector<double> vec;
    auto normed = double_rot.quaternion(); // Make sure out quaternion is normalized
    normed.normalize();
    auto normed_vec = normed.vec();
    for(int i = 0; i < 3; i++) // do the vector first, then the scalar
    {
      vec.push_back(normed_vec[i]);
    }

    vec.push_back(normed.w());

    ret_obj = py::cast<std::vector<double> >(vec);
  }
  else if(type == 'f')
  {
    std::vector<float> vec;
    auto normed = float_rot.quaternion(); // Make sure out quaternion is normalized
    normed.normalize();
    auto normed_vec = normed.vec();
    for(int i = 0; i < 3; i++) // do the vector first, then the scalar
    {
      vec.push_back(normed_vec[i]);
    }

    vec.push_back(normed.w());

    ret_obj = py::cast<std::vector<float> >(vec);
  }
  return ret_obj;
}

py::object
PyRotation::
rodrigues()
{
  py::object rod;
  if(type == 'f')
  {
    rod = py::cast<Eigen::Matrix<float,3,1>>(float_rot.rodrigues());
  }
  else if(type == 'd')
  {
    rod = py::cast<Eigen::Matrix<double,3,1>>(double_rot.rodrigues());
  }

  return rod;
}

py::object
PyRotation::
rotate_vector(py::object vec_obj)
{
  py::object ret_obj;
  if(type == 'f')
  {
    Eigen::Matrix< float, 3, -1 > vec = vec_obj.cast< Eigen::Matrix< float, 3, -1 > >();
    Eigen::Matrix< float, 3, 1> vec3(vec.data());
    Eigen::Matrix< float, 3, 1 > ret_vec = float_rot * vec3;
    ret_obj = py::cast< Eigen::Matrix< float, 3, 1 > >(ret_vec);
  }
  else if(type == 'd')
  {
    Eigen::Matrix< double, 3, -1 > vec = vec_obj.cast< Eigen::Matrix< double, 3, -1 > >();
    Eigen::Matrix< double, 3, 1> vec3(vec.data());
    Eigen::Matrix< double, 3, 1 > ret_vec = double_rot * vec3;
    ret_obj = py::cast< Eigen::Matrix< double, 3, 1 > >(ret_vec);
  }

  return ret_obj;
}

py::list
PyRotation::
get_yaw_pitch_roll()
{
  py::list ypr = py::list();
  if(type == 'f')
  {
    float y,p,r;
    float_rot.get_yaw_pitch_roll(y,p,r);
    ypr.append(y);
    ypr.append(p);
    ypr.append(r);
  }
  else if(type == 'd')
  {
    double y,p,r;
    double_rot.get_yaw_pitch_roll(y,p,r);
    ypr.append(y);
    ypr.append(p);
    ypr.append(r);
  }

  return ypr;
}

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

class PyDescriptorBase
{

  public:

    virtual ~PyDescriptorBase() = default;

    virtual size_t get_size() {return 0;};
    virtual size_t get_num_bytes() {return 0;};
    virtual size_t sum() {return -1;};

    virtual void set_slice(py::slice slice, py::object val_obj) {};
    virtual py::object get_slice(py::slice slice) {return py::none();};

    virtual std::vector<double> as_double() { return std::vector<double>();};
    virtual std::vector<unsigned char> as_bytes() { return std::vector<unsigned char>();};

};

class PyDescriptorD
: public PyDescriptorBase
{

  kwiver::vital::descriptor_dynamic<double> desc;

  public:

    PyDescriptorD(size_t len) 
      : desc (kwiver::vital::descriptor_dynamic<double>(len)) 
      {};

    size_t get_size() { return desc.size(); };
    size_t get_num_bytes() { return desc.num_bytes(); };

    size_t sum()
    {
      double* data = desc.raw_data();
      size_t sum = 0;
      for (size_t idx = 0; idx < desc.size(); idx++)
      {
        sum += data[idx];
      }
      return sum;
    }

    std::vector<double> as_double() { return desc.as_double();};
    std::vector<unsigned char> as_bytes() { return desc.as_bytes();};

    void set_slice(py::slice slice, py::object val_obj)
    {
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      double* data = desc.raw_data();

      try
      {
        double val = val_obj.cast<double>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val;
        }
      }
      catch(...)
      {
        std::vector<double> val = val_obj.cast<std::vector<double>>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val[idx];
        }
      }
    };

    py::object get_slice(py::slice slice)
    {
      std::vector<double> ret_vec;
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      double* data = desc.raw_data();

      for (size_t idx = start; idx < stop; idx+=step)
      {
        ret_vec.push_back(data[idx]);
      }
      return py::cast<std::vector<double>> (ret_vec);
    }

};

class PyDescriptorF
: public PyDescriptorBase
{

  kwiver::vital::descriptor_dynamic<float> desc;

  public:

    PyDescriptorF(size_t len)
      : desc(kwiver::vital::descriptor_dynamic<float>(len))
      {};

    size_t get_size() { return desc.size(); };
    size_t get_num_bytes() { return desc.num_bytes(); };

    size_t sum()
    {
      float* data = desc.raw_data();
      size_t sum = 0;
      for (size_t idx = 0; idx < desc.size(); idx++)
      {
        sum += data[idx];
      }
      return sum;
    }

    std::vector<double> as_double() { return desc.as_double();};
    std::vector<unsigned char> as_bytes() { return desc.as_bytes();};

    void set_slice(py::slice slice, py::object val_obj)
    {
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      float* data = desc.raw_data();

      try
      {
        float val = val_obj.cast<float>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val;
        }
      }
      catch(...)
      {
        std::vector<float> val = val_obj.cast<std::vector<float>>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val[idx];
        }
      }
    };

    py::object get_slice(py::slice slice)
    {
      std::vector<float> ret_vec;
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      float* data = desc.raw_data();

      for (size_t idx = start; idx < stop; idx+=step)
      {
        ret_vec.push_back(data[idx]);
      }
      return py::cast<std::vector<float>> (ret_vec);
    }
};

std::shared_ptr<PyDescriptorBase>
new_descriptor(size_t len, char ctype)
{
  std::shared_ptr<PyDescriptorBase> retVal;
  if(ctype == 'd')
  {
    retVal = std::shared_ptr<PyDescriptorBase>(new PyDescriptorD(len));
  }
  else if(ctype == 'f')
  {
    retVal = std::shared_ptr<PyDescriptorBase>(new PyDescriptorF(len));
  }
  return retVal;
}


