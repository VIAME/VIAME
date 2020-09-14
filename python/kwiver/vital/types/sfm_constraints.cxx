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

#include <vital/types/sfm_constraints.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( sfm_constraints, m )
{
  py::class_< kv::sfm_constraints,
              std::shared_ptr< kv::sfm_constraints > >( m, "SFMConstraints" )
  .def( py::init<>() )
  .def( py::init< kv::sfm_constraints const& >() )
  .def( py::init< kv::metadata_map_sptr, kv::local_geo_cs const& >() )
  .def_property( "metadata",
                  &kv::sfm_constraints::get_metadata,
                  &kv::sfm_constraints::set_metadata )
  .def_property( "local_geo_cs",
                  &kv::sfm_constraints::get_local_geo_cs,
                  &kv::sfm_constraints::set_local_geo_cs )
  .def( "get_camera_position_prior_local",
        ( bool ( kv::sfm_constraints::* ) ( kv::frame_id_t, kv::vector_3d& ) const )
        &kv::sfm_constraints::get_camera_position_prior_local )
  .def( "get_camera_position_prior_local",
        ( bool ( kv::sfm_constraints::* ) ( kv::frame_id_t, kv::rotation_d& ) const )
        &kv::sfm_constraints::get_camera_position_prior_local )
  .def( "get_camera_position_priors", &kv::sfm_constraints::get_camera_position_priors )
  .def( "store_image_size", &kv::sfm_constraints::store_image_size )
  // Ints are immutable in python so simple bindings to get_image_width/height, get_focal_length_prior wont be sufficient
  // lambda functions will return a tuple of the results as per https://pybind11.readthedocs.io/en/stable/faq.html#limitations-involving-reference-arguments
  .def( "get_image_width", [](kv::sfm_constraints const &self, kv::frame_id_t fid, int &image_width )
  {
      bool found = self.get_image_width(fid, image_width);
      return std::make_tuple(found, image_width);
  })
  .def( "get_image_height", [](kv::sfm_constraints const &self, kv::frame_id_t fid, int &image_height )
  {
      bool found = self.get_image_height(fid, image_height);
      return std::make_tuple(found, image_height);
  })
  .def( "get_focal_length_prior", [](kv::sfm_constraints const &self, kv::frame_id_t fid, float &focal_length )
  {
      bool found = self.get_focal_length_prior(fid, focal_length);
      return std::make_tuple(found, focal_length);
  })
  ;
}
