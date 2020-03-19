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

#include <vital/types/camera_intrinsics.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these methods can be overriden in C++
PYBIND11_MODULE( camera_intrinsics_helpers, m )
{
  m.def( "clone", [] ( const kv::camera_intrinsics& self )
  {
    return self.clone();
  });

  m.def( "focal_length", [] ( const kv::camera_intrinsics& self )
  {
    return self.focal_length();
  });

  m.def( "principal_point", [] ( const kv::camera_intrinsics& self )
  {
    return self.principal_point();
  });

  m.def( "aspect_ratio", [] ( const kv::camera_intrinsics& self )
  {
    return self.aspect_ratio();
  });

  m.def( "skew", [] ( const kv::camera_intrinsics& self )
  {
    return self.skew();
  });

  m.def( "image_width", [] ( const kv::camera_intrinsics& self )
  {
    return self.image_width();
  });

  m.def( "image_height", [] ( const kv::camera_intrinsics& self )
  {
    return self.image_height();
  });

  m.def( "dist_coeffs", [] ( const kv::camera_intrinsics& self )
  {
    return self.dist_coeffs();
  });

  m.def( "as_matrix", [] ( const kv::camera_intrinsics& self )
  {
    return self.as_matrix();
  });

  m.def( "map", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.map( norm_pt );
  });

  m.def( "map", [] ( const kv::camera_intrinsics& self, const kv::vector_3d& norm_hpt )
  {
    return self.map( norm_hpt );
  });

  m.def( "unmap", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.unmap( norm_pt );
  });

  m.def( "distort", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.distort( norm_pt );
  });

  m.def( "undistort", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& dist_pt )
  {
    return self.undistort( dist_pt );
  });

  m.def( "is_map_valid", [] ( const kv::camera_intrinsics& self, const kv::vector_2d& norm_pt )
  {
    return self.is_map_valid( norm_pt );
  });

  m.def( "is_map_valid", [] ( const kv::camera_intrinsics& self, const kv::vector_3d& norm_hpt )
  {
    return self.is_map_valid( norm_hpt );
  });

}
