// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/integrate_depth_maps_trampoline.txx>
#include <python/kwiver/vital/algo/integrate_depth_maps.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void integrate_depth_maps(py::module &m)
{
  py::class_< kwiver::vital::algo::integrate_depth_maps,
              std::shared_ptr<kwiver::vital::algo::integrate_depth_maps>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::integrate_depth_maps>,
              integrate_depth_maps_trampoline<> >(m, "IntegrateDepthMaps")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::integrate_depth_maps::static_type_name)
    .def("integrate",
        (void (kwiver::vital::algo::integrate_depth_maps::*)
          (vector_3d const&, vector_3d const&, std::vector<image_container_sptr> const&,
           std::vector<image_container_sptr> const&, std::vector<camera_perspective_sptr> const&,
           image_container_sptr&, vector_3d&) const)
        &kwiver::vital::algo::integrate_depth_maps::integrate)
    .def("integrate",
        (void (kwiver::vital::algo::integrate_depth_maps::*)
          (vector_3d const&, vector_3d const&, std::vector<image_container_sptr> const&,
           std::vector<camera_perspective_sptr> const&, image_container_sptr&, vector_3d&) const)
        &kwiver::vital::algo::integrate_depth_maps::integrate);
}
}
}
}
