/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include <stdexcept>

#include <vital/vital_types.h>
#include <vital/types/homography.h>
#include <vital/types/homography_f2f.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>

namespace kv = kwiver::vital;
namespace py = pybind11;

using f2f_homography = kv::f2f_homography;

PYBIND11_MODULE(homography_f2f, m)
{
  // This should wrap all of f2f_homography except for the (templated)
  // constructor directly from an Eigen::Matrix and the copy
  // constructor
  py::class_<f2f_homography, std::shared_ptr<f2f_homography>>(m, "F2FHomography")
    .def(py::init<kv::homography_sptr const&, kv::frame_id_t, kv::frame_id_t>())
    .def(py::init<kv::frame_id_t>())
    .def_property_readonly("homography", &f2f_homography::homography)
    .def_property_readonly("from_id", &f2f_homography::from_id)
    .def_property_readonly("to_id", &f2f_homography::to_id)
    .def("inverse", &f2f_homography::inverse)
    .def("__mul__", &f2f_homography::operator*)
    .def("get",
	 [] (f2f_homography const& self, int r, int c)
	 {
	   auto m = self.homography()->matrix();
	   if(0 <= r && r < m.rows() && 0 <= c && c < m.cols())
	   {
	     return m(r, c);
	   }
	   throw std::out_of_range("Tried to perform get() out of bounds");
	 },
	 "Convenience method that returns the underlying coefficient"
	 " at the given row and column")
    ;
}
