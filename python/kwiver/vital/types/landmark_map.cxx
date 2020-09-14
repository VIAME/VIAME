/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

#include <vital/types/landmark_map.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace kv = kwiver::vital;
typedef kv::simple_landmark_map s_landmark_map;
typedef std::map< kv::landmark_id_t, kv::landmark_sptr > map_landmark_t;
using namespace kwiver::vital;

class landmark_map_trampoline
: public landmark_map
{
public:
  using landmark_map::landmark_map;
  size_t size() const override;
  map_landmark_t landmarks() const override;
};


PYBIND11_MODULE(landmark_map, m)
{
  py::bind_map< map_landmark_t >(m, "LandmarkDict");

  py::class_< landmark_map, std::shared_ptr< landmark_map >, landmark_map_trampoline >(m, "LandmarkMap")
  .def(py::init())
  .def("size", &landmark_map::size)
  .def("landmarks", &landmark_map::landmarks, py::return_value_policy::reference)

  .def("__repr__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
      classname = self.__class__.__name__
      retval = '<%s at %s>' % (classname, hex(id(self)))
      )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
  })

  .def("__str__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
      classname = self.__class__.__name__
      retval = '<%s>' % (classname)
      )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
  });


  py::class_< s_landmark_map, landmark_map, std::shared_ptr< s_landmark_map > >(m, "SimpleLandmarkMap")
  .def(py::init<>())
  .def(py::init< map_landmark_t >());
}

size_t
landmark_map_trampoline
::size() const
{
  PYBIND11_OVERLOAD_PURE(
    size_t,
    landmark_map,
    size,
  );
}

map_landmark_t
landmark_map_trampoline
::landmarks() const
{
  PYBIND11_OVERLOAD_PURE(
    map_landmark_t,
    landmark_map,
    landmarks,
  )
}
