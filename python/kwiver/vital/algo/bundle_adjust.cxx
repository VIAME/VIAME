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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <python/kwiver/vital/algo/trampoline/bundle_adjust_trampoline.txx>
#include <python/kwiver/vital/algo/bundle_adjust.h>

namespace py = pybind11;

void bundle_adjust(py::module &m)
{
  py::class_< kwiver::vital::algo::bundle_adjust,
              std::shared_ptr<kwiver::vital::algo::bundle_adjust>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::bundle_adjust>,
              bundle_adjust_trampoline<> >(m, "BundleAdjust")
    .def(py::init())
    .def_static("static_type_name", &kwiver::vital::algo::bundle_adjust::static_type_name)
    .def("optimize", static_cast<void (kwiver::vital::algo::bundle_adjust::*)
                                  (kwiver::vital::camera_map_sptr&,
                                   kwiver::vital::landmark_map_sptr&,
                                   kwiver::vital::feature_track_set_sptr,
                                   kwiver::vital::sfm_constraints_sptr) const>
                     (&kwiver::vital::algo::bundle_adjust::optimize))
    .def("optimize", static_cast<void (kwiver::vital::algo::bundle_adjust::*)
                                 (kwiver::vital::simple_camera_perspective_map&,
                                  kwiver::vital::landmark_map::map_landmark_t&,
                                  kwiver::vital::feature_track_set_sptr,
                                  const std::set<kwiver::vital::frame_id_t>&,
                                  const std::set<kwiver::vital::landmark_id_t>&,
                                  kwiver::vital::sfm_constraints_sptr) const>
                     (&kwiver::vital::algo::bundle_adjust::optimize))
    .def("set_callback", &kwiver::vital::algo::bundle_adjust::set_callback);
}
