// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/optimize_cameras_trampoline.txx>
#include <python/kwiver/vital/algo/optimize_cameras.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void optimize_cameras(py::module &m)
{
  py::class_< kwiver::vital::algo::optimize_cameras,
              std::shared_ptr<kwiver::vital::algo::optimize_cameras>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::optimize_cameras>,
              optimize_cameras_trampoline<> >(m, "OptimizeCameras")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::optimize_cameras::static_type_name)
    .def("optimize",
        ( void
          ( kwiver::vital::algo::optimize_cameras::* )
          ( kwiver::vital::camera_map_sptr&,
            kwiver::vital::feature_track_set_sptr,
            kwiver::vital::landmark_map_sptr,
            kwiver::vital::sfm_constraints_sptr ) const
        )
        &kwiver::vital::algo::optimize_cameras::optimize)
    .def("optimize",
        ( void
          ( kwiver::vital::algo::optimize_cameras::* )
          ( kwiver::vital::camera_perspective_sptr&,
            std::vector< kwiver::vital::feature_sptr > const& features,
            std::vector< kwiver::vital::landmark_sptr > const& landmarks,
            kwiver::vital::sfm_constraints_sptr ) const
        )
        &kwiver::vital::algo::optimize_cameras::optimize);
}
}
}
}
