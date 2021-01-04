// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <python/kwiver/vital/algo/trampoline/bundle_adjust_trampoline.txx>
#include <python/kwiver/vital/algo/bundle_adjust.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
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
}
}
}
