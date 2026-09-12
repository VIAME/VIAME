// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/camera_intrinsics.h>

#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <sstream>

namespace py = pybind11;
namespace kv = kwiver::vital;

using vector_t = kv::simple_camera_intrinsics::vector_t;

class camera_intrinsics_trampoline
  : public kv::camera_intrinsics
{
public:
  using kv::camera_intrinsics::camera_intrinsics;

  kv::camera_intrinsics_sptr clone() const override;
  double focal_length() const override;
  kv::vector_2d principal_point() const override;
  double aspect_ratio() const override;
  double skew() const override;

  size_t image_width() const override;
  size_t image_height() const override;
  std::vector< double > dist_coeffs() const override;
  kv::matrix_3x3d as_matrix() const override;
  kv::vector_2d map( const kv::vector_2d& norm_pt ) const override;
  kv::vector_2d map( const kv::vector_3d& norm_hpt ) const override;
  kv::vector_2d unmap( const kv::vector_2d& norm_pt ) const override;
  kv::vector_2d distort( const kv::vector_2d& norm_pt ) const override;
  kv::vector_2d undistort( const kv::vector_2d& dist_pt ) const override;
  bool is_map_valid( const kv::vector_2d& norm_pt ) const override;
  bool is_map_valid( const kv::vector_3d& norm_hpt ) const override;
};

PYBIND11_MODULE( camera_intrinsics, m )
{
  py::class_< kv::camera_intrinsics,
    std::shared_ptr< kv::camera_intrinsics >,
    camera_intrinsics_trampoline >( m, "CameraIntrinsics" )
    .def( py::init<>() )
    .def( "clone",           &kv::camera_intrinsics::clone )
    .def( "focal_length",    &kv::camera_intrinsics::focal_length )
    .def( "principal_point", &kv::camera_intrinsics::principal_point )
    .def( "aspect_ratio",    &kv::camera_intrinsics::aspect_ratio )
    .def( "skew",            &kv::camera_intrinsics::skew )
    .def( "image_width",     &kv::camera_intrinsics::image_width )
    .def( "image_height",    &kv::camera_intrinsics::image_height )
    .def( "dist_coeffs",     &kv::camera_intrinsics::dist_coeffs )
    .def( "as_matrix",       &kv::camera_intrinsics::as_matrix )
    // map overloads
    .def(
      "map",
      ( kv::vector_2d ( kv::camera_intrinsics::* )(
        const kv::vector_2d& ) const ) &
      kv::camera_intrinsics::map, py::arg( "norm_pt" ) )
    .def(
      "map",
      ( kv::vector_2d ( kv::camera_intrinsics::* )(
        const kv::vector_3d& ) const ) &
      kv::camera_intrinsics::map, py::arg( "norm_pt" ) )
    .def(
      "unmap",
      &kv::camera_intrinsics::unmap,
      py::arg( "norm_pt" ) )
    .def(
      "distort",
      &kv::camera_intrinsics::distort,
      py::arg( "norm_pt" ) )
    .def(
      "undistort",
      &kv::camera_intrinsics::undistort,
      py::arg( "norm_pt" ) )
    // is_map_valid overloads
    .def(
      "is_map_valid",
      ( bool ( kv::camera_intrinsics::* )( const kv::vector_2d& ) const ) &
      kv::camera_intrinsics::is_map_valid, py::arg( "norm_pt" ) )
    .def(
      "is_map_valid",
      ( bool ( kv::camera_intrinsics::* )( const kv::vector_3d& ) const ) &
      kv::camera_intrinsics::is_map_valid, py::arg( "norm_pt" ) )

    .def(
      "__str__", []( const kv::camera_intrinsics& self ){
        std::stringstream s;
        s << self;
        return s.str();
      } )
  ;

  py::class_< kv::simple_camera_intrinsics,
    kv::camera_intrinsics,
    std::shared_ptr< kv::simple_camera_intrinsics > >(
    m,
    "SimpleCameraIntrinsics" )
    .def( py::init<>() )
    .def(
      py::init< const double, const kv::vector_2d&, const double, const double,
        const vector_t, const size_t, const size_t >(),
      py::arg( "focal_length" ), py::arg( "principal_point" ),
      py::arg( "aspect_ratio" ) = 1.0, py::arg( "skew" ) = 0.0,
      py::arg( "dist_coeffs" ) = vector_t(),  py::arg( "image_width" ) = 0,
      py::arg( "image_height" ) = 0 )
    .def( py::init< const kv::camera_intrinsics& >(), py::arg( "base" ) )
    .def(
      py::init< const kv::matrix_3x3d&, const vector_t& >(),
      py::arg( "K" ), py::arg( "d" ) = vector_t() )
    .def_static(
      "from_string", []( const std::string& s ){
        kv::simple_camera_intrinsics self;
        std::istringstream ss( s );
        ss >> self;
        return self;
      } )
    .def(
      "max_distort_radius",
      &kv::simple_camera_intrinsics::max_distort_radius )
    .def(
      "get_max_distort_radius_sq",
      &kv::simple_camera_intrinsics::get_max_distort_radius_sq )
    .def(
      "set_focal_length",
      &kv::simple_camera_intrinsics::set_focal_length,
      py::arg( "focal_length" ) )
    .def(
      "set_principal_point",
      &kv::simple_camera_intrinsics::set_principal_point, py::arg( "pp" ) )
    .def(
      "set_aspect_ratio",
      &kv::simple_camera_intrinsics::set_aspect_ratio,
      py::arg( "aspect_ratio" ) )
    .def(
      "set_skew",
      &kv::simple_camera_intrinsics::set_skew, py::arg( "skew" ) )
    .def(
      "set_image_width",
      &kv::simple_camera_intrinsics::set_image_width, py::arg( "width" ) )
    .def(
      "set_image_height",
      &kv::simple_camera_intrinsics::set_image_height, py::arg( "height" ) )
    .def(
      "set_dist_coeffs",
      &kv::simple_camera_intrinsics::set_dist_coeffs, py::arg( "d" ) )
    .def_static(
      "max_distort_radius_sq",
      &kv::simple_camera_intrinsics::max_distort_radius_sq, py::arg( "a" ),
      py::arg( "b" ), py::arg( "c" ) )
  ;
}

kv::camera_intrinsics_sptr
camera_intrinsics_trampoline
::clone() const
{
  auto self = py::cast( this );

  auto cloned = self.attr("clone")();

  auto python_keep_alive = std::make_shared< py::object >( cloned );

  auto ptr = cloned.cast< camera_intrinsics_trampoline* >();

  return std::shared_ptr< kv::camera_intrinsics >( python_keep_alive, ptr );
}

double
camera_intrinsics_trampoline
::focal_length() const
{
  PYBIND11_OVERLOAD_PURE(
    double,
    kv::camera_intrinsics,
    focal_length,
  );
}

kv::vector_2d
camera_intrinsics_trampoline
::principal_point() const
{
  PYBIND11_OVERLOAD_PURE(
    kv::vector_2d,
    kv::camera_intrinsics,
    principal_point,
  );
}

double
camera_intrinsics_trampoline
::aspect_ratio() const
{
  PYBIND11_OVERLOAD_PURE(
    double,
    kv::camera_intrinsics,
    aspect_ratio,
  );
}

double
camera_intrinsics_trampoline
::skew() const
{
  PYBIND11_OVERLOAD_PURE(
    double,
    kv::camera_intrinsics,
    skew,
  );
}

size_t
camera_intrinsics_trampoline
::image_width() const
{
  PYBIND11_OVERLOAD_PURE(
    size_t,
    kv::camera_intrinsics,
    image_width,
  );
}

size_t
camera_intrinsics_trampoline
::image_height() const
{
  PYBIND11_OVERLOAD_PURE(
    size_t,
    kv::camera_intrinsics,
    image_height,
  );
}

std::vector< double >
camera_intrinsics_trampoline
::dist_coeffs() const
{
  PYBIND11_OVERLOAD(
    std::vector< double >,
    kv::camera_intrinsics,
    dist_coeffs,
  );
}

kv::matrix_3x3d
camera_intrinsics_trampoline
::as_matrix() const
{
  PYBIND11_OVERLOAD(
    kv::matrix_3x3d,
    kv::camera_intrinsics,
    as_matrix,
  );
}

kv::vector_2d
camera_intrinsics_trampoline
::map( const kv::vector_2d& norm_pt ) const
{
  PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_intrinsics,
    map,
    norm_pt
  );
}

kv::vector_2d
camera_intrinsics_trampoline
::map( const kv::vector_3d& norm_hpt ) const
{
  PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_intrinsics,
    map,
    norm_hpt
  );
}

kv::vector_2d
camera_intrinsics_trampoline
::unmap( const kv::vector_2d& norm_pt ) const
{
  PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_intrinsics,
    unmap,
    norm_pt
  );
}

kv::vector_2d
camera_intrinsics_trampoline
::distort( const kv::vector_2d& norm_pt ) const
{
  PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_intrinsics,
    distort,
    norm_pt
  );
}

kv::vector_2d
camera_intrinsics_trampoline
::undistort( const kv::vector_2d& dist_pt ) const
{
  PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_intrinsics,
    undistort,
    dist_pt
  );
}

bool
camera_intrinsics_trampoline
::is_map_valid( const kv::vector_2d& norm_pt ) const
{
  PYBIND11_OVERLOAD(
    bool,
    kv::camera_intrinsics,
    is_map_valid,
    norm_pt
  );
}

bool
camera_intrinsics_trampoline
::is_map_valid( const kv::vector_3d& norm_hpt ) const
{
  PYBIND11_OVERLOAD(
    bool,
    kv::camera_intrinsics,
    is_map_valid,
    norm_hpt
  );
}
