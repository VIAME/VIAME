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

#include <vital/types/track_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;

std::shared_ptr<kwiver::vital::track>
get_track(std::shared_ptr<kwiver::vital::track_set> &self, uint64_t id)
{
  auto track = self->get_track(id);
  if(!track)
  {
    throw py::index_error("Track does not exist in set");
  }

  return track;
}

PYBIND11_MODULE(track_set, m)
{
  py::class_<kwiver::vital::track_set, std::shared_ptr<kwiver::vital::track_set>>(m, "TrackSet")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::track>>>())
  .def("all_frame_ids", &kwiver::vital::track_set::all_frame_ids)
  .def("get_track", &get_track,
    py::arg("id"))
  .def("first_frame", &kwiver::vital::track_set::first_frame)
  .def("last_frame", &kwiver::vital::track_set::last_frame)
  .def("size", &kwiver::vital::track_set::size)
  .def("tracks", &kwiver::vital::track_set::tracks)
  .def("__len__", &kwiver::vital::track_set::size)
  ;
}
