/*ckwg +29
 * Copyright 2017, 2020 by Kitware, Inc.
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

#include <vital/types/camera_perspective.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;

class camera_perspective_trampoline
  : public kv::camera_perspective
{
public:
  using kv::camera_perspective::camera_perspective;

  virtual kv::camera_sptr clone() const override;
  virtual kv::vector_3d center() const override;
  virtual kv::vector_3d translation() const override;
  virtual kv::covariance_3d center_covar() const override;
  virtual kv::rotation_d rotation() const override;
  virtual kv::camera_intrinsics_sptr intrinsics() const override;
  virtual unsigned int image_width() const override;
  virtual unsigned int image_height() const override;

  virtual kv::camera_perspective_sptr clone_look_at(
    const kv::vector_3d &stare_point,
    const kv::vector_3d &up_direction ) const override;

  virtual kv::matrix_3x4d as_matrix() const override;
  virtual kv::vector_2d project( const kv::vector_3d& pt ) const override;
  virtual double depth( const kv::vector_3d& pt ) const override;
};

PYBIND11_MODULE( camera_perspective, m )
{
  py::class_< kv::camera_perspective,
              std::shared_ptr< kv::camera_perspective >,
              kv::camera,
              camera_perspective_trampoline >( m, "CameraPerspective" )
  .def( py::init<>() )
  .def( "center",        &kv::camera_perspective::center )
  .def( "translation",   &kv::camera_perspective::translation )
  .def( "center_covar",  &kv::camera_perspective::center_covar )
  .def( "rotation",      &kv::camera_perspective::rotation )
  .def( "intrinsics",    &kv::camera_perspective::intrinsics )
  .def( "image_width",   &kv::camera_perspective::image_width )
  .def( "image_height",  &kv::camera_perspective::image_height )
  .def( "as_matrix",     &kv::camera_perspective::as_matrix )
  .def( "pose_matrix",   &kv::camera_perspective::pose_matrix )
  .def( "project",       &kv::camera_perspective::project )
  .def( "depth",         &kv::camera_perspective::depth )

  .def( "__str__", [] ( const kv::camera_perspective& self )
  {
    std::stringstream s;
    s << self;
    return s.str();
  })
  ;

  py::class_< kv::simple_camera_perspective,
              std::shared_ptr< kv::simple_camera_perspective >,
              kv::camera_perspective >( m, "SimpleCameraPerspective" )
  .def( py::init<>() )
  .def( py::init< const kv::vector_3d&, const kv::rotation_d&, kv::camera_intrinsics_sptr >(),
        py::arg( "center" ), py::arg( "rotation" ),
        py::arg( "intrinsics" ) =  kv::camera_intrinsics_sptr() )
  .def( py::init< const kv::vector_3d&, const kv::rotation_d&, const kv::camera_intrinsics& >() )
  .def( py::init< const kv::camera_perspective& >() )
  .def_static( "from_string", [] ( const std::string& s )
  {
    kv::simple_camera_perspective self;
    std::istringstream ss( s );
    ss >> self;
    return self;
  })
  // Don't need to rebind methods in base class, just new methods
  .def( "get_center",       &kv::simple_camera_perspective::get_center )
  .def( "get_center_covar", &kv::simple_camera_perspective::get_center_covar )
  .def( "get_rotation",     &kv::simple_camera_perspective::get_rotation )
  .def( "get_intrinsics",   &kv::simple_camera_perspective::get_intrinsics )
  .def( "set_center",       &kv::simple_camera_perspective::set_center )
  .def( "set_translation",  &kv::simple_camera_perspective::set_translation )
  .def( "set_center_covar", &kv::simple_camera_perspective::set_center_covar )
  .def( "set_rotation",     &kv::simple_camera_perspective::set_rotation )
  .def( "set_intrinsics",   &kv::simple_camera_perspective::set_intrinsics )
  .def( "look_at",          &kv::simple_camera_perspective::look_at,
         py::arg( "stare_point" ), py::arg( "up_direction" ) = kv::vector_3d::UnitZ() )

  ;
}

kv::camera_sptr
camera_perspective_trampoline
::clone() const
{

  auto self = py::cast(this);

  auto cloned = self.attr("clone")();

  auto python_keep_alive = std::make_shared<py::object>(cloned);

  auto ptr = cloned.cast<camera_perspective_trampoline*>();

  return std::shared_ptr<kv::camera_perspective>(python_keep_alive, ptr);
}

kv::camera_perspective_sptr
camera_perspective_trampoline
::clone_look_at(
  const kv::vector_3d &stare_point,
  const kv::vector_3d &up_direction = kv::vector_3d::UnitZ() ) const
{
  auto self = py::cast(this);

  auto cloned = self.attr("clone_look_at")(stare_point,up_direction);

  auto python_keep_alive = std::make_shared<py::object>(cloned);

  auto ptr = cloned.cast<camera_perspective_trampoline*>();

  return std::shared_ptr<kv::camera_perspective>(python_keep_alive, ptr);
}

kv::vector_3d
camera_perspective_trampoline
::center() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_3d,
    kv::camera_perspective,
    center,
  );
}

kv::vector_3d
camera_perspective_trampoline
::translation() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_3d,
    kv::camera_perspective,
    translation,
  );
}

kv::covariance_3d
camera_perspective_trampoline
::center_covar() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::covariance_3d,
    kv::camera_perspective,
    center_covar,
  );
}

kv::rotation_d
camera_perspective_trampoline
::rotation() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::rotation_d,
    kv::camera_perspective,
    rotation,
  );
}

kv::camera_intrinsics_sptr
camera_perspective_trampoline
::intrinsics() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::camera_intrinsics_sptr,
    kv::camera_perspective,
    intrinsics,
  );
}

unsigned int
camera_perspective_trampoline
::image_width() const
{
  VITAL_PYBIND11_OVERLOAD(
    unsigned int,
    kv::camera_perspective,
    image_width,
  );
}

unsigned int
camera_perspective_trampoline
::image_height() const
{
  VITAL_PYBIND11_OVERLOAD(
    unsigned int,
    kv::camera_perspective,
    image_height,
  );
}

kv::matrix_3x4d
camera_perspective_trampoline
::as_matrix() const
{
  VITAL_PYBIND11_OVERLOAD(
    kv::matrix_3x4d,
    kv::camera_perspective,
    as_matrix,
  );
}

kv::vector_2d
camera_perspective_trampoline
::project( const kv::vector_3d& pt ) const
{
  VITAL_PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_perspective,
    project,
    pt
  );
}

double
camera_perspective_trampoline
::depth( const kv::vector_3d& pt ) const
{
  VITAL_PYBIND11_OVERLOAD(
    double,
    kv::camera_perspective,
    depth,
    pt
  );
}
