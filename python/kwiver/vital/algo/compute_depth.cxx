// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <python/kwiver/vital/algo/trampoline/compute_depth_trampoline.txx>
#include <python/kwiver/vital/algo/compute_depth.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void compute_depth(py::module &m)
{
  py::class_< kwiver::vital::algo::compute_depth,
              std::shared_ptr<kwiver::vital::algo::compute_depth>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::compute_depth>,
              compute_depth_trampoline<> >( m, "ComputeDepth" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::compute_depth::static_type_name)
    .def("compute", static_cast<kwiver::vital::image_container_sptr
                                (kwiver::vital::algo::compute_depth::*)
                                (std::vector<kwiver::vital::image_container_sptr> const&,
                                 std::vector<kwiver::vital::camera_perspective_sptr> const&,
                                 double, double,
                                 unsigned int,
                                 vital::bounding_box<int> const&,
                                 std::vector<kwiver::vital::image_container_sptr> const&) const>
                    (&kwiver::vital::algo::compute_depth::compute))
    .def("compute", static_cast<kwiver::vital::image_container_sptr
                                (kwiver::vital::algo::compute_depth::*)
                                (std::vector<kwiver::vital::image_container_sptr> const&,
                                 std::vector<kwiver::vital::camera_perspective_sptr> const&,
                                 double, double,
                                 unsigned int,
                                 vital::bounding_box<int> const&,
                                 kwiver::vital::image_container_sptr&,
                                 std::vector<kwiver::vital::image_container_sptr> const&) const>
                    (&kwiver::vital::algo::compute_depth::compute))
    .def("set_callback",
        &kwiver::vital::algo::compute_depth::set_callback);
}
}
}
}
