/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#include <vital/types/camera_rpc.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;


class camera_rpc_trampoline
  :public kv::camera_rpc
{
public:
  using kv::camera_rpc::camera_rpc;

  kv::camera_sptr clone() const override;
  kv::rpc_matrix rpc_coeffs() const override;
  kv::vector_3d world_scale() const override;
  kv::vector_3d world_offset() const override;
  kv::vector_2d image_scale() const override;
  kv::vector_2d image_offset() const override;
  unsigned int image_width() const override;
  unsigned int image_height() const override;
  kv::vector_2d project( const kv::vector_3d& pt ) const override;
  kv::vector_3d back_project( const kv::vector_2d& image_pt, double elev ) const override;
  void jacobian( const kv::vector_3d& pt,
                 kv::matrix_2x2d& J,
                 kv::vector_2d& norm_pt ) const override;
};

class camera_rpc_publicist
  :public kv::camera_rpc
{
public:

  using kv::camera_rpc::jacobian;

};

typedef
void ( kv::camera_rpc::*jacob_fun_ptr_t )
( const kv::vector_3d&, kv::matrix_2x2d&, kv::vector_2d& ) const;

PYBIND11_MODULE( camera_rpc, m )
{
  py::module::import( "kwiver.vital.types.camera" );

  py::class_< kv::camera_rpc,
              std::shared_ptr< kv::camera_rpc >,
              kv::camera,
              camera_rpc_trampoline >( m, "CameraRPC" )
  .def( py::init<>() )
  .def_static( "power_vector", kv::camera_rpc::power_vector )
  .def( "clone",        &kv::camera_rpc::clone )
  .def( "rpc_coeffs",   &kv::camera_rpc::rpc_coeffs )
  .def( "world_scale",  &kv::camera_rpc::world_scale )
  .def( "world_offset", &kv::camera_rpc::world_offset )
  .def( "image_scale",  &kv::camera_rpc::image_scale )
  .def( "image_offset", &kv::camera_rpc::image_offset )
  .def( "image_width",  &kv::camera_rpc::image_width )
  .def( "image_height", &kv::camera_rpc::image_height )
  .def( "project",      &kv::camera_rpc::project )
  .def( "back_project", &kv::camera_rpc::back_project )
  // TODO: Jacobian returns an eigen matrix by reference, both of which are things pybind does poorly.
  .def( "jacobian",     static_cast<jacob_fun_ptr_t>(&camera_rpc_publicist::jacobian))
  ;

  py::class_< kv::simple_camera_rpc,
              std::shared_ptr< kv::simple_camera_rpc >,
              kv::camera_rpc >( m, "SimpleCameraRPC" )
  .def( py::init<>() )
  .def( py::init< kv::vector_3d&, kv::vector_3d&,
                  kv::vector_2d&, kv::vector_2d&,
                  kv::rpc_matrix&, unsigned int,
                  unsigned int >(),
                  py::arg( "world_scale" ), py::arg( "world_offset" ),
                  py::arg( "image_scale" ), py::arg( "image_offset" ),
                  py::arg( "rpc_coeffs" ),  py::arg( "image_width" ) = 0,
                  py::arg( "image_height" ) = 0 )
  .def( py::init< const kv::camera_rpc& >() )
  .def( "set_rpc_coeffs",   &kv::simple_camera_rpc::set_rpc_coeffs )
  .def( "set_world_scale",  &kv::simple_camera_rpc::set_world_scale )
  .def( "set_world_offset", &kv::simple_camera_rpc::set_world_offset )
  .def( "set_image_scale",  &kv::simple_camera_rpc::set_image_scale )
  .def( "set_image_offset", &kv::simple_camera_rpc::set_image_offset )
  .def( "set_image_width",  &kv::simple_camera_rpc::set_image_width )
  .def( "set_image_height", &kv::simple_camera_rpc::set_image_height )
  ;
}

kv::camera_sptr
camera_rpc_trampoline
::clone() const
{
  auto self = py::cast(this);

  auto cloned = self.attr("clone")();

  auto python_keep_alive = std::make_shared<py::object>(cloned);

  auto ptr = cloned.cast<camera_rpc_trampoline*>();

  return std::shared_ptr<kv::camera_rpc>(python_keep_alive, ptr);
}

kv::rpc_matrix
camera_rpc_trampoline
::rpc_coeffs() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::rpc_matrix,
    kv::camera_rpc,
    rpc_coeffs,
  );
}

kv::vector_3d
camera_rpc_trampoline
::world_scale() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_3d,
    kv::camera_rpc,
    world_scale,
  );
}

kv::vector_3d
camera_rpc_trampoline
::world_offset() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_3d,
    kv::camera_rpc,
    world_offset,
  );
}

kv::vector_2d
camera_rpc_trampoline
::image_scale() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_2d,
    kv::camera_rpc,
    image_scale,
  );
}

kv::vector_2d
camera_rpc_trampoline
::image_offset() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_2d,
    kv::camera_rpc,
    image_offset,
  );
}

unsigned int
camera_rpc_trampoline
::image_width() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    unsigned int,
    kv::camera_rpc,
    image_width,
  );
}

unsigned int
camera_rpc_trampoline
::image_height() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    unsigned int,
    kv::camera_rpc,
    image_height,
  );
}

kv::vector_2d
camera_rpc_trampoline
::project( const kv::vector_3d& pt ) const
{
  VITAL_PYBIND11_OVERLOAD(
    kv::vector_2d,
    kv::camera_rpc,
    project,
    pt
  );
}

kv::vector_3d
camera_rpc_trampoline
::back_project( const kv::vector_2d& image_pt, double elev ) const
{
  VITAL_PYBIND11_OVERLOAD(
    kv::vector_3d,
    kv::camera_rpc,
    back_project,
    image_pt,
    elev
  );
}

void
camera_rpc_trampoline
::jacobian( const kv::vector_3d& pt,
               kv::matrix_2x2d& J,
               kv::vector_2d& norm_pt ) const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    void,
    kv::camera_rpc,
    jacobian,
    pt,
    J,
    norm_pt
  );
}
