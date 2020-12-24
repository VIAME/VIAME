// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/homography.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py=pybind11;
namespace kwiver {
namespace vital  {
namespace python {

class PyHomographyBase
{

  public:

    virtual ~PyHomographyBase() = default;

    virtual char get_type() {return '0';};
    virtual py::object get_matrix() {return py::none();};

    virtual py::object inverse() {return py::none();};
    virtual py::object map(py::object) {return py::none();};
    virtual py::object normalize() {return py::none();};

    bool operator==(std::shared_ptr<PyHomographyBase> &);
    bool operator!=(std::shared_ptr<PyHomographyBase> &other) {return !(*this == other);};
    virtual std::shared_ptr<PyHomographyBase> operator*(std::shared_ptr<PyHomographyBase> &) {return std::shared_ptr<PyHomographyBase>();};
};

class PyHomographyD
: public PyHomographyBase
{
  kwiver::vital::homography_<double> homog;

  public:

    PyHomographyD() {};
    PyHomographyD(Eigen::Matrix<double, 3, 3> mat) : homog(kwiver::vital::homography_<double>(mat)) {};
    PyHomographyD(kwiver::vital::homography_<double> mat) : homog(mat) {};

    char get_type() {return 'd';};
    py::object get_matrix() { return py::cast<Eigen::Matrix<double, 3, 3>>(homog.get_matrix()); };

    py::object inverse();
    py::object map(py::object);
    py::object normalize();

    std::shared_ptr<PyHomographyBase> operator*(std::shared_ptr<PyHomographyBase> &);
};

class PyHomographyF
: public PyHomographyBase
{
  kwiver::vital::homography_<float> homog;

  public:

    PyHomographyF() {};
    PyHomographyF(Eigen::Matrix<float, 3, 3> mat) : homog(kwiver::vital::homography_<float>(mat)) {};
    PyHomographyF(kwiver::vital::homography_<float> mat) : homog(mat) {};

    char get_type() {return 'f';};
    py::object get_matrix() { return py::cast<Eigen::Matrix<float, 3, 3>>(homog.get_matrix()); };

    py::object inverse();
    py::object map(py::object);
    py::object normalize();

    std::shared_ptr<PyHomographyBase> operator*(std::shared_ptr<PyHomographyBase> &);
};

py::object
PyHomographyD
::inverse()
{
  auto new_homog = this->homog.inverse();
  auto new_matrix = new_homog->matrix();
  return py::cast<Eigen::Matrix<double, 3, 3>>(new_matrix);
}

py::object
PyHomographyD
::map(py::object p_obj)
{
  auto p = p_obj.cast<Eigen::Matrix<double,2,1>>();
  auto new_matrix = this->homog.map_point(p);
  return py::cast<Eigen::Matrix<double, 2, 1>>(new_matrix);
}

py::object
PyHomographyD
::normalize()
{
  auto new_homog = this->homog.normalize();
  auto new_matrix = new_homog->matrix();
  return py::cast<Eigen::Matrix<double, 3, 3>>(new_matrix);
}

std::shared_ptr<PyHomographyBase>
PyHomographyD::
operator*(std::shared_ptr<PyHomographyBase> &other)
{
  auto mat = other->get_matrix().cast<Eigen::Matrix<double,3,3>>(); //this part is a bit silly, could be done better
  auto other_homog = kwiver::vital::homography_<double>(mat);
  return std::shared_ptr<PyHomographyBase>(new PyHomographyD(this->homog*other_homog));
}

py::object
PyHomographyF
::inverse()
{
  auto new_homog = this->homog.inverse();
  auto new_matrix = new_homog->matrix();
  return py::cast<Eigen::Matrix<double, 3, 3>>(new_matrix);
}

py::object
PyHomographyF
::map(py::object p_obj)
{
  auto p = p_obj.cast<Eigen::Matrix<float,2,1>>();
  auto new_matrix = this->homog.map_point(p);
  return py::cast<Eigen::Matrix<float, 2, 1>>(new_matrix);
}

py::object
PyHomographyF
::normalize()
{
  auto new_homog = this->homog.normalize();
  auto new_matrix = new_homog->matrix();
  return py::cast<Eigen::Matrix<double, 3, 3>>(new_matrix);
}

std::shared_ptr<PyHomographyBase>
PyHomographyF::
operator*(std::shared_ptr<PyHomographyBase> &other)
{
  auto mat = other->get_matrix().cast<Eigen::Matrix<float,3,3>>(); //this part is a bit silly, could be done better
  auto other_homog = kwiver::vital::homography_<float>(mat);
  return std::shared_ptr<PyHomographyBase>(new PyHomographyF(this->homog*other_homog));
}

bool
PyHomographyBase::
operator==(std::shared_ptr<PyHomographyBase> &other)
{
  if(this->get_type() == 'd')
  {
    auto this_mat = this->get_matrix().cast<Eigen::Matrix<double, 3, 3>>();
    auto other_mat = other->get_matrix().cast<Eigen::Matrix<double, 3, 3>>();
    return this_mat.isApprox(other_mat);
  }
  else if(this->get_type() == 'f')
  {
    auto this_mat = this->get_matrix().cast<Eigen::Matrix<float, 3, 3>>();
    auto other_mat = other->get_matrix().cast<Eigen::Matrix<float, 3, 3>>();
    return this_mat.isApprox(other_mat, 0.001);
  }

  return false;
}

std::shared_ptr<PyHomographyBase>
new_homography(char ctype)
{
  std::shared_ptr<PyHomographyBase> retVal;
  if(ctype == 'd')
  {
    retVal = std::shared_ptr<PyHomographyBase>(new PyHomographyD());
  }
  else if(ctype == 'f')
  {
    retVal = std::shared_ptr<PyHomographyBase>(new PyHomographyF());
  }
  return retVal;
}

std::shared_ptr<PyHomographyBase>
new_homography_from_matrix(py::object data_obj, char ctype)
{
  std::shared_ptr<PyHomographyBase> retVal;
  if(ctype == 'd')
  {
    auto data = data_obj.cast<Eigen::Matrix<double, 3,3>>();
    retVal = std::shared_ptr<PyHomographyBase>(new PyHomographyD(data));
  }
  else if(ctype == 'f')
  {
    auto data = data_obj.cast<Eigen::Matrix<float, 3,3>>();
    retVal = std::shared_ptr<PyHomographyBase>(new PyHomographyF(data));
  }
  return retVal;
}

std::shared_ptr<PyHomographyBase>
new_random_homography(char ctype)
{
  py::object data_obj;
  if(ctype == 'd')
  {
    data_obj = py::cast<Eigen::Matrix<double, 3, 3>>(Eigen::MatrixXd::Random(3,3));
  }
  else if(ctype == 'f')
  {
    data_obj = py::cast<Eigen::Matrix<float, 3, 3>>(Eigen::MatrixXf::Random(3,3));
  }
  return new_homography_from_matrix(data_obj, ctype);
}

}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(homography, m)
{
  py::class_<PyHomographyBase, std::shared_ptr<PyHomographyBase>>(m, "Homography")
  .def(py::init(&new_homography),
    py::arg("type")='d')
  .def_static("from_matrix", &new_homography_from_matrix,
    py::arg("data"), py::arg("type")='d')
  .def_static("random", &new_random_homography,
    py::arg("type")='d')
  .def_property_readonly("type_name", &PyHomographyBase::get_type)
  .def("as_matrix", &PyHomographyBase::get_matrix)
  .def("inverse", &PyHomographyBase::inverse)
  .def("map", &PyHomographyBase::map,
    py::arg("point"))
  .def("normalize", &PyHomographyBase::normalize)
  .def("__eq__", &PyHomographyBase::operator==)
  .def("__ne__", &PyHomographyBase::operator!=)
  .def("__mul__", &PyHomographyBase::operator*)
  ;
}
