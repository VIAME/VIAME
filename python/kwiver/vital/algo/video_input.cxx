/*ckwg +29
 * Copyright 2015-2018 by Kitware, Inc.
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
#include <python/kwiver/vital/algo/trampoline/video_input_trampoline.txx>
#include <python/kwiver/vital/algo/video_input.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void video_input(py::module &m)
{
  py::class_< kwiver::vital::algo::video_input,
              std::shared_ptr<kwiver::vital::algo::video_input>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::video_input>,
              video_input_trampoline<> >(m, "VideoInput")
  .def(py::init())
  .def_readonly_static("HAS_EOV",
    &kwiver::vital::algo::video_input::HAS_EOV)
  .def_readonly_static("HAS_FRAME_NUMBERS",
    &kwiver::vital::algo::video_input::HAS_FRAME_NUMBERS)
  .def_readonly_static("HAS_FRAME_TIME",
    &kwiver::vital::algo::video_input::HAS_FRAME_TIME)
  .def_readonly_static("HAS_FRAME_DATA",
    &kwiver::vital::algo::video_input::HAS_FRAME_DATA)
  .def_readonly_static("HAS_FRAME_RATE",
    &kwiver::vital::algo::video_input::HAS_FRAME_RATE)
  .def_readonly_static("HAS_ABSOLUTE_FRAME_TIME",
    &kwiver::vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME)
  .def_readonly_static("HAS_METADATA",
    &kwiver::vital::algo::video_input::HAS_METADATA)
  .def_readonly_static("HAS_TIMEOUT",
    &kwiver::vital::algo::video_input::HAS_TIMEOUT)
  .def_readonly_static("IS_SEEKABLE",
    &kwiver::vital::algo::video_input::IS_SEEKABLE)
  .def_static("static_type_name",
    &kwiver::vital::algo::video_input::static_type_name)
  .def("open",
    &kwiver::vital::algo::video_input::open)
  .def("close",
    &kwiver::vital::algo::video_input::close)
  .def("end_of_video",
    &kwiver::vital::algo::video_input::end_of_video)
  .def("good",
    &kwiver::vital::algo::video_input::good)
  .def("seekable",
    &kwiver::vital::algo::video_input::seekable)
  .def("num_frames",
    &kwiver::vital::algo::video_input::num_frames)
  .def("next_frame",
    &kwiver::vital::algo::video_input::next_frame)
  .def("seek_frame",
    &kwiver::vital::algo::video_input::seek_frame)
  .def("frame_timestamp",
    &kwiver::vital::algo::video_input::frame_timestamp)
  .def("frame_image",
    &kwiver::vital::algo::video_input::frame_image)
  .def("frame_metadata",
    &kwiver::vital::algo::video_input::frame_metadata)
  .def("metadata_map",
    &kwiver::vital::algo::video_input::metadata_map)
  .def("frame_rate",
    &kwiver::vital::algo::video_input::frame_rate)
  .def("get_implementation_capabilities",
    &kwiver::vital::algo::video_input::get_implementation_capabilities);
}
