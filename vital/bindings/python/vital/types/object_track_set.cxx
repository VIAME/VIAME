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

#include <vital/types/object_track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

typedef kwiver::vital::object_track_state obj_track_state;
typedef kwiver::vital::object_track_set obj_track_set;

namespace py = pybind11;

std::shared_ptr<kwiver::vital::track>
get_track(std::shared_ptr<obj_track_set> &self, uint64_t id)
{
  auto track = self->get_track(id);
  if(!track)
  {
    throw py::index_error("Track does not exist in set");
  }

  return track;
}

PYBIND11_MODULE(object_track_set, m)
{
  py::class_<obj_track_state, kwiver::vital::track_state, std::shared_ptr<obj_track_state>>(m, "ObjectTrackState")
  .def(py::init<int64_t, int64_t, std::shared_ptr<kwiver::vital::detected_object>>())
  .def_property_readonly("frame_id", &kwiver::vital::track_state::frame)
  .def_property_readonly("time_usec", &kwiver::vital::object_track_state::time)
  .def_readwrite("detection", &obj_track_state::detection)
  ;

  py::class_<obj_track_set, kwiver::vital::track_set, std::shared_ptr<obj_track_set>>(m, "ObjectTrackSet")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::track>>>())
  .def("all_frame_ids", &obj_track_set::all_frame_ids)
  .def("get_track", &get_track,
    py::arg("id"))
  .def("first_frame", &obj_track_set::first_frame)
  .def("last_frame", &obj_track_set::last_frame)
  .def("size", &obj_track_set::size)
  .def("tracks", &obj_track_set::tracks)
  .def("__len__", &obj_track_set::size)
  ;
}
