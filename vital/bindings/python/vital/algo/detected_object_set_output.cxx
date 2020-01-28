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
#include <pybind11/pybind11.h>
#include <vital/bindings/python/vital/algo/trampoline/detected_object_set_output_trampoline.txx>
#include <vital/bindings/python/vital/algo/detected_object_set_output.h>

namespace py = pybind11;

using doso = kwiver::vital::algo::detected_object_set_output;

class file_wrapper
{
  doso* obj;
  std::ostream& stream()
  {
    // Use a subclass to expose and access a protected member.
    // See also https://github.com/pybind/pybind11/issues/991#issuecomment-321266887
    struct doso_pub : doso { using doso::stream; };
    return (obj->*&doso_pub::stream)();
  }
public:
  file_wrapper(doso& obj)
    : obj(&obj) {}
  void write(std::string const& s)
  {
    stream().write(s.data(), s.size());
  }
  void flush()
  {
    stream().flush();
  }
};

void detected_object_set_output(py::module &m)
{
  py::class_< doso,
              std::shared_ptr<doso>,
              kwiver::vital::algorithm_def<doso>,
              detected_object_set_output_trampoline<> >(m, "DetectedObjectSetOutput")
    .def(py::init())
    .def_static("static_type_name", &doso::static_type_name)
    .def("write_set", &doso::write_set)
    .def("complete", &doso::complete)
    .def("open", &doso::open)
    .def("close", &doso::close)
    .def_property_readonly("_stream", [](doso& self) { return new file_wrapper(self); });

  py::class_<file_wrapper>(m, "DetectedObjectSetOutputFileWrapper")
    .def("write", &file_wrapper::write)
    .def("flush", &file_wrapper::flush)
    ;
}
