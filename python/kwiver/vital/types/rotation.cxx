// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/rotation.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

namespace py = pybind11;
namespace kv = kwiver::vital;


template < typename T >
py::tuple
rot_get_yaw_pitch_roll( kv::rotation_< T > const& self )
{
  T y, p, r;
  self.get_yaw_pitch_roll(y, p, r);
  return py::make_tuple(y, p, r);
}

template < typename T >
std::vector< kv::rotation_< T > >
rot_interpolated_rotations( kv::rotation_< T > const& A, kv::rotation_< T > const& B, size_t n )
{
  std::vector< kv::rotation_< T > > ret;
  interpolated_rotations(A, B, n, ret);
  return ret;
}



// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< typename T >
void declare_rotation( py::module &m,
                       std::string const& class_typestr,
                       std::string const& dtype )
{
  using Class = kv::rotation_< T >;
  const std::string pyclass_name = std::string( "Rotation" ) + class_typestr;

  py::class_< Class, std::shared_ptr< Class > >( m, pyclass_name.c_str() )
    .def( py::init() )
    .def( py::init< const kv::rotation_< float >& >() )
    .def( py::init< const kv::rotation_< double >& >() )
    .def( py::init< const Eigen::Matrix< T, 4, 1 >& >() )
    .def( py::init< const Eigen::Matrix< T, 3, 1 >& >() )
    .def( py::init< T, const Eigen::Matrix< T, 3, 1 >& >() )
    .def( py::init< const T&, const T&, const T& >() )
    .def( py::init< const Eigen::Matrix< T, 3, 3 >& >() )
    .def( "matrix", &Class::matrix )
    .def( "axis", &Class::axis )
    .def( "angle", &Class::angle )
    .def( "angle_from", [] ( Class const& self, Class const& other)
    {
      return self.quaternion().angularDistance(other.quaternion());
    })
    .def( "quaternion", [] ( Class const& self )
    {
      std::vector< T > vec;
      auto normed = self.quaternion();
      normed.normalize();
      auto normed_vec = normed.vec();
      for ( int i = 0; i < 3; i++ )
      {
        vec.push_back( normed_vec[i] );
      }
      vec.push_back( normed.w() );

      return vec;
    })
    .def( "rodrigues", &Class::rodrigues )
    .def( "yaw_pitch_roll", &rot_get_yaw_pitch_roll< T > )
    .def( "inverse", &Class::inverse )
    .def( "__mul__", [] ( const Class& self, const Class& other )
    {
      return self * other;
    })
    .def( "__mul__", [] ( const Class& self, const Eigen::Matrix< T, 3, 1 >& rhs )
    {
      return self * rhs;
    })
    .def( "__eq__", [] ( const Class& self, const Class& other )
    {
      return self == other;
    })
    .def( "__ne__", [] ( const Class& self, const Class& other )
    {
      return self != other;
    })
    .def_property_readonly( "type_name", [ dtype ] ( Class const& self )
    {
      return dtype;
    })
    ;

  m.def( "interpolate_rotation", &kv::interpolate_rotation< T > );
  m.def( "interpolated_rotations", &rot_interpolated_rotations< T > );


}

using namespace kwiver::vital::python;

PYBIND11_MODULE(rotation, m)
{
  declare_rotation< float > ( m, "F", "f" );
  declare_rotation< double >( m, "D", "d" );
}
