// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
