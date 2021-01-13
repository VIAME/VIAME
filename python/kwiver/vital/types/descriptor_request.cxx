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

#include <vital/types/descriptor_request.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE(descriptor_request, m)
{
  py::class_<kwiver::vital::descriptor_request,
    std::shared_ptr<kwiver::vital::descriptor_request>>(m, "DescriptorRequest")
  .def(py::init<>())
  // Expose the member variables with getters and setters
  .def_property("id", &kv::descriptor_request::id, &kv::descriptor_request::set_id)
  .def_property("spatial_regions", &kv::descriptor_request::spatial_regions, &kv::descriptor_request::set_spatial_regions)
  .def_property("image_data", &kv::descriptor_request::image_data, &kv::descriptor_request::set_image_data)
  .def_property("data_location", &kv::descriptor_request::data_location, &kv::descriptor_request::set_data_location)
  // There aren't separate setters for each of the temporal bounds
  // member variables, so we'll expose the corresponding functions like normal
  .def("temporal_lower_bound", &kv::descriptor_request::temporal_lower_bound)
  .def("temporal_upper_bound", &kv::descriptor_request::temporal_upper_bound)
  .def("set_temporal_bounds", &kv::descriptor_request::set_temporal_bounds)
  ;
}
