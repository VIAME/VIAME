/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include <vital/types/track.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;

py::object
track_find_state(kwiver::vital::track &self, int64_t frame_id)
{
  auto frame_itr = self.find(frame_id);
  if(frame_itr == self.end())
  {
    throw py::index_error();
  }
  return py::cast<std::shared_ptr<kwiver::vital::track_state>>(*frame_itr);
}

PYBIND11_MODULE(track, m)
{
  py::class_<kwiver::vital::track_state, std::shared_ptr<kwiver::vital::track_state>>(m, "TrackState")
  .def(py::init<int64_t>())
  .def_property_readonly("frame_id", &kwiver::vital::track_state::frame)
  ;

  py::class_<kwiver::vital::track, std::shared_ptr<kwiver::vital::track>>(m, "Track")
  .def(py::init([](int64_t id)
    {
      auto track = kwiver::vital::track::create();
      track->set_id(id);
      return track;
    }),
    py::arg("id")=0)
  .def("all_frame_ids", &kwiver::vital::track::all_frame_ids)
  .def("append", [](kwiver::vital::track &self, std::shared_ptr<kwiver::vital::track_state> track_state)
    {
      return self.append(track_state);
    })
  .def("append", [](kwiver::vital::track &self, kwiver::vital::track &track)
    {
      return self.append(track);
    })
  .def("find_state", &track_find_state)
  .def("__iter__", [](const kwiver::vital::track &self)
    {
      return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0,1>())
  .def("__len__", &kwiver::vital::track::size)
  .def("__getitem__", &track_find_state)
  .def_property("id", &kwiver::vital::track::id, &kwiver::vital::track::set_id)
  .def_property_readonly("size", &kwiver::vital::track::size)
  .def_property_readonly("is_empty", &kwiver::vital::track::empty)
  .def_property_readonly("first_frame", &kwiver::vital::track::first_frame)
  .def_property_readonly("last_frame", &kwiver::vital::track::last_frame)

  ;
}
