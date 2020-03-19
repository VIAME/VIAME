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

#include <vital/types/query_result.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

// Note that this is unlike query_result::descriptors().
// This returns a copy of the track_descriptor_set, while
// query_result::descriptors() returns a pointer to
// the internal track_descriptor_set. The returned set cannot
// be used to modify by reference
kv::track_descriptor_set
query_result_get_descriptors(const kv::query_result& self)
{
  kv::track_descriptor_set_sptr descs_sptr = self.descriptors();
  if (!descs_sptr)
  {
    return kv::track_descriptor_set();
  }
  return *descs_sptr;
}

// Pybind wont allow track_descriptor_sets to be passed around with smart_pointers
// We'll make a smart_ptr to a copy of the track_descriptor_set passed in
void
query_result_set_descriptors(kv::query_result& self, const kv::track_descriptor_set& tdset)
{
  // Note that we're making a copy of a vector of pointers.
  // Elements of the vector can still be modified by reference on the python side
  self.set_descriptors(kv::track_descriptor_set_sptr(new kv::track_descriptor_set(tdset)));
}

PYBIND11_MODULE(query_result, m)
{
  py::class_<kwiver::vital::query_result, std::shared_ptr<kwiver::vital::query_result>>(m, "QueryResult")
  .def(py::init<>())
  // Expose the member variables with getters and setters
  .def_property("query_id", &kv::query_result::query_id, &kv::query_result::set_query_id)
  .def_property("stream_id", &kv::query_result::stream_id, &kv::query_result::set_stream_id)
  .def_property("instance_id", &kv::query_result::instance_id, &kv::query_result::set_instance_id)
  .def_property("relevancy_score", &kv::query_result::relevancy_score, &kv::query_result::set_relevancy_score)
  .def_property("location", &kv::query_result::location, &kv::query_result::set_location)
  .def_property("tracks", &kv::query_result::tracks, &kv::query_result::set_tracks)
  .def_property("descriptors", &query_result_get_descriptors, &query_result_set_descriptors)
  .def_property("image_data", &kv::query_result::image_data, &kv::query_result::set_image_data)
  // There aren't separate setters for the start_time and end_time
  // member variables, so we'll expose the corresponding functions like normal
  .def("start_time", &kv::query_result::start_time)
  .def("end_time", &kv::query_result::end_time)
  .def("set_temporal_bounds", &kv::query_result::set_temporal_bounds)
  ;
}
