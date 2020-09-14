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

#include <vital/types/camera_perspective.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kwiver::vital::camera_perspective cam_p;
// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( camera_perspective_helpers, m )
{
  m.def( "call_clone", [] ( const kv::camera_perspective &self)
  {
    return self.clone();
  });

  m.def( "call_center", [] ( const kv::camera_perspective &self )
  {
    return self.center();
  });

  m.def( "call_translation", [] ( const kv::camera_perspective &self )
  {
    return self.translation();
  });

  m.def( "call_center_covar", [] ( const kv::camera_perspective &self )
  {
    return self.center_covar();
  });

  m.def( "call_rotation", [] ( const kv::camera_perspective &self )
  {
    return self.rotation();
  });

  m.def( "call_intrinsics", [] ( const kv::camera_perspective &self )
  {
    return self.intrinsics();
  });

  m.def( "call_image_width", [] ( const kv::camera_perspective &self )
  {
    return self.image_width();
  });

  m.def( "call_image_height", [] ( const kv::camera_perspective &self )
  {
    return self.image_height();
  });

  m.def( "call_clone_look_at", [] ( const kv::camera_perspective &self,
                               const kv::vector_3d &stare_pt,
                               const kv::vector_3d &up_direc )
  {
    return self.clone_look_at(stare_pt,up_direc);
  });

  m.def( "call_as_matrix", [] ( const kv::camera_perspective &self )
  {
    return self.as_matrix();
  });

  m.def( "call_project", [] ( const kv::camera_perspective &self, kv::vector_3d &pt )
  {
    return self.project(pt);
  });

  m.def( "call_depth", [] ( const kv::camera_perspective &self, kv::vector_3d &pt )
  {
    return self.depth(pt);
  });
}
