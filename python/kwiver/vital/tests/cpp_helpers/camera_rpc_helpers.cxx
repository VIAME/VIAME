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

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kwiver::vital::camera_rpc crpc;

// Helpers to call pure virtual functions from base reference.
// We'll use these to test that these camera methods can be overriden in C++
PYBIND11_MODULE( camera_rpc_helpers, m )
{
  m.def( "call_clone", [] ( const kv::camera_rpc &self)
  {
    return self.clone();
  });

  m.def( "call_rpc_coeffs", [] ( const kv::camera_rpc &self )
  {
    return self.rpc_coeffs();
  });

  m.def( "call_world_scale", [] ( const kv::camera_rpc &self )
  {
    return self.world_scale();
  });

  m.def( "call_world_offset", [] ( const kv::camera_rpc &self )
  {
    return self.world_offset();
  });

  m.def( "call_image_scale", [] ( const kv::camera_rpc &self )
  {
    return self.image_scale();
  });

  m.def( "call_image_offset", [] ( const kv::camera_rpc &self )
  {
    return self.image_offset();
  });

  m.def( "call_image_width", [] ( const kv::camera_rpc &self )
  {
    return self.image_width();
  });

  m.def( "call_image_height", [] ( const kv::camera_rpc &self )
  {
    return self.image_height();
  });

  m.def( "call_project", [] ( const kv::camera_rpc &self,
                               const kv::vector_3d &pt )
  {
    return self.project(pt);
  });

  m.def( "call_back_project", [] ( const kv::camera_rpc &self,
                                         kv::vector_2d &pt,
                                         double elev )
  {
    return self.back_project(pt, elev);
  });
}
