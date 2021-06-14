// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/resection_camera_trampoline.txx>
#include <python/kwiver/vital/algo/resection_camera.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void resection_camera(py::module &m)
{
  py::class_< kwiver::vital::algo::resection_camera,
              std::shared_ptr<kwiver::vital::algo::resection_camera>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::resection_camera>,
              resection_camera_trampoline<> >( m, "ResectionCamera" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::resection_camera::static_type_name)
    .def("resection",
         static_cast<kwiver::vital::camera_perspective_sptr
                     (kwiver::vital::algo::resection_camera::*)
                     (std::vector<kwiver::vital::vector_2d> const&,
                      std::vector<kwiver::vital::vector_3d> const&,
                      kwiver::vital::camera_intrinsics_sptr,
                      std::vector<bool>*) const>
         (&kwiver::vital::algo::resection_camera::resection))
    .def("resection",
         static_cast<kwiver::vital::camera_perspective_sptr
                     (kwiver::vital::algo::resection_camera::*)
                     (kwiver::vital::frame_id_t,
                      kwiver::vital::landmark_map_sptr,
                      kwiver::vital::feature_track_set_sptr,
                      unsigned width, unsigned height,
                      std::unordered_set< landmark_id_t >*) const>
         (&kwiver::vital::algo::resection_camera::resection))
    .def("resection",
         static_cast<kwiver::vital::camera_perspective_sptr
                     (kwiver::vital::algo::resection_camera::*)
                     (kwiver::vital::frame_id_t,
                      kwiver::vital::landmark_map_sptr,
                      kwiver::vital::feature_track_set_sptr,
                      kwiver::vital::camera_intrinsics_sptr,
                      std::unordered_set< landmark_id_t >*) const>
         (&kwiver::vital::algo::resection_camera::resection));
}
}
}
}
