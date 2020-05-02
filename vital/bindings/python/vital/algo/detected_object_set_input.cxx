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
#include <utility>

#include <pybind11/pybind11.h>
#include <vital/bindings/python/vital/algo/trampoline/detected_object_set_input_trampoline.txx>
#include <vital/bindings/python/vital/algo/detected_object_set_input.h>

namespace py = pybind11;

using dosi = kwiver::vital::algo::detected_object_set_input;

void detected_object_set_input(py::module &m)
{
  py::class_< dosi,
              std::shared_ptr<dosi>,
              kwiver::vital::algorithm_def<dosi>,
              detected_object_set_input_trampoline<> >(m, "DetectedObjectSetInput")
    .def(py::init())
    .def_static("static_type_name", &dosi::static_type_name)
    .def("read_set",
	 [](dosi& self) {
	   std::pair<kwiver::vital::detected_object_set_sptr, std::string> result;
	   bool has_result = self.read_set(result.first, result.second);
	   return has_result ? py::cast(result) : py::cast(nullptr);
	 },
	 R"(Return a pair of the next DetectedObjectSet and the corresponding
file name, or None if the input is exhausted)")
    .def("read_set_by_path",
	 [](dosi& self, std::string path) {
	   kwiver::vital::detected_object_set_sptr result;
	   self.read_set(result, path);
	   return result;
	 },
	 R"(Return the DetectedObjectSet for the provided file name)")
    .def("open", &dosi::open)
    .def("close", &dosi::close)
    ;
}
