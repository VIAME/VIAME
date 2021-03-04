/*ckwg +29
 * Copyright 2019-2020 by Kitware, Inc.
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

#include <vital/types/homography_f2w.h>

#include <pybind11/pybind11.h>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE( homography_f2w, m )
{
  py::class_< kv::f2w_homography, std::shared_ptr< kv::f2w_homography > >( m, "F2WHomography" )
  .def( py::init< kv::frame_id_t const >() )
  .def( py::init< kv::homography_sptr const&, kv::frame_id_t const >() )
  .def( py::init< kv::f2w_homography const& >() )
  .def_property_readonly( "homography", &kv::f2w_homography::homography )
  .def_property_readonly( "frame_id", &kv::f2w_homography::frame_id )
  .def( "get",
   [] ( kv::f2w_homography const& self, int r, int c )
   {
     auto m = self.homography()->matrix();
     if( 0 <= r && r < m.rows() && 0 <= c && c < m.cols() )
     {
       return m( r, c );
     }
     throw std::out_of_range( "Tried to perform get() out of bounds" );
   },
   "Convenience method that returns the underlying coefficient"
   " at the given row and column" )
  ;
}
