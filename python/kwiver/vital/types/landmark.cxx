// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/landmark.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vital/util/demangle.h>

#include <string>


namespace py = pybind11;
namespace kv = kwiver::vital;
using namespace kwiver::vital;

// std::shared_ptr<PyLandmarkBase>
// new_landmark(py::object loc_obj, py::object scale_obj, char ctype)
// {
//   std::shared_ptr<PyLandmarkBase> retVal;
//   if(ctype == 'd')
//   {
//     // Get our arguments, taking care of default cases
//     Eigen::Matrix<double, 3, 1> loc = Eigen::Matrix<double, 3, 1>();
//     loc << 0,0,0;
//     if(!loc_obj.is(py::none()))
//     {
//       loc = loc_obj.cast<Eigen::Matrix<double, 3, 1>>();
//     }

//     double scale = 1.;
//     if(!scale_obj.is(py::none()))
//     {
//       scale = scale_obj.cast<double>();
//     }

//     retVal = std::shared_ptr<PyLandmarkBase>(new PyLandmarkd(loc, scale));
//   }
//   else if(ctype == 'f')
//   {
//     // Get our arguments, taking care of default cases
//     Eigen::Matrix<float, 3, 1> loc = Eigen::Matrix<float, 3, 1>();
//     loc << 0,0,0;
//     if(!loc_obj.is(py::none()))
//     {
//       loc = loc_obj.cast<Eigen::Matrix<float, 3, 1>>();
//     }

//     float scale = 1.;
//     if(!scale_obj.is(py::none()))
//     {
//       scale = scale_obj.cast<float>();
//     }

//     retVal = std::shared_ptr<PyLandmarkBase>(new PyLandmarkf(loc, scale));
//   }

//   return retVal;
// }

template <typename T>
void reg_landmark(py::module &m, std::string &&type_str)
{
  using Class =  kv::landmark_< T >;
  std::string py_class_name = std::string("Landmark") + type_str;
  py::class_< Class, kv::landmark, std::shared_ptr< Class >  >(m, py_class_name.c_str())
  .def(py::init())
  .def(py::init< Eigen::Matrix< T, 3, 1 > const&, T >(), py::arg("loc"), py::arg("scale") = 1 )
  .def(py::init< kv::landmark const& >())
  .def_property_readonly("data_type", ([](Class const &self)
  {
    return demangle( self.data_type().name() );
  }))
  .def_property("loc", &Class::get_loc, &Class::set_loc)
  .def_property("scale", &Class::get_scale, &Class::set_scale)
  .def_property("normal", &Class::get_normal, &Class::set_normal)
  .def_property("covariance", &Class::get_covar, &Class::set_covar)
  .def_property("color", &Class::get_color, &Class::set_color)
  .def_property("observations", &Class::get_observations, &Class::set_observations)
  .def_property("cos_obs_angle", &Class::get_cos_obs_angle, &Class::set_cos_observation_angle)
  ;

}

class landmark_trampoline
: public kv::landmark
{
public:
  using kv::landmark::landmark;
  kv::landmark_sptr clone() const override;
  std::type_info const& data_type() const override;
  kv::vector_3d loc() const override;
  double scale() const override;
  kv::vector_3d normal() const override;
  kv::covariance_3d covar() const override;
  kv::rgb_color color() const override;
  unsigned observations() const override;
  double cos_obs_angle() const override;
};


PYBIND11_MODULE(landmark, m)
{
  py::class_<kv::landmark, std::shared_ptr< kv::landmark >, landmark_trampoline >(m, "Landmark")
  .def(py::init<>())
  .def("data_type",     &kv::landmark::data_type)
  .def("loc",           &kv::landmark::loc)
  .def("scale",         &kv::landmark::scale)
  .def("normal",        &kv::landmark::normal)
  .def("covar",         &kv::landmark::covar)
  .def("color",         &kv::landmark::color)
  .def("observations",   &kv::landmark::observations)
  .def("cos_obs_angle",  &kv::landmark::cos_obs_angle)
  ;

  reg_landmark<double>(m, "D");
  reg_landmark<float>(m, "F");
}

kv::landmark_sptr
landmark_trampoline
::clone() const
{
  auto self = py::cast(this);

  auto cloned = self.attr("clone")();

  auto python_keep_alive = std::make_shared<py::object>(cloned);

  auto ptr = cloned.cast<landmark_trampoline*>();

  return std::shared_ptr<kv::landmark>(python_keep_alive, ptr);
}

std::type_info const &
landmark_trampoline
::data_type() const
{
  PYBIND11_OVERLOAD_PURE(
    std::type_info const &,
    kv::landmark,
    data_type,
  );
}

kv::vector_3d
landmark_trampoline
::loc() const
{
  PYBIND11_OVERLOAD_PURE(
    kv::vector_3d,
    kv::landmark,
    loc,
  );
}

double
landmark_trampoline
::scale() const
{
  PYBIND11_OVERLOAD_PURE(
    double,
    kv::landmark,
    scale,
  );
}

kv::vector_3d
landmark_trampoline
::normal() const
{
  PYBIND11_OVERLOAD_PURE(
    kv::vector_3d,
    kv::landmark,
    normal,
  );
}

kv::covariance_3d
landmark_trampoline
::covar() const
{
  PYBIND11_OVERLOAD_PURE(
    kv::covariance_3d,
    kv::landmark,
    covar,
  );
}

kv::rgb_color
landmark_trampoline
::color() const
{
  PYBIND11_OVERLOAD_PURE(
    kv::rgb_color,
    kv::landmark,
    color,
  );
}

unsigned
landmark_trampoline
::observations() const
{
  PYBIND11_OVERLOAD_PURE(
    unsigned,
    kv::landmark,
    observations,
  );
}

double
landmark_trampoline
::cos_obs_angle() const
{
  PYBIND11_OVERLOAD_PURE(
    double,
    kv::landmark,
    cos_obs_angle,
  );
}
