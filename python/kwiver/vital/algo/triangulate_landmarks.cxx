// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/triangulate_landmarks_trampoline.txx>
#include <python/kwiver/vital/algo/triangulate_landmarks.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void triangulate_landmarks(py::module &m)
{
  py::class_< kwiver::vital::algo::triangulate_landmarks,
              std::shared_ptr<kwiver::vital::algo::triangulate_landmarks>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::triangulate_landmarks>,
              triangulate_landmarks_trampoline<> >(m, "TriangulateLandmarks")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::triangulate_landmarks::static_type_name)
    .def("triangulate",
         ( void (kwiver::vital::algo::triangulate_landmarks::*)
           ( kwiver::vital::camera_map_sptr,
             kwiver::vital::feature_track_set_sptr,
             kwiver::vital::landmark_map_sptr&
           ) const
         )
         &kwiver::vital::algo::triangulate_landmarks::triangulate)
    .def("triangulate",
         ( void (kwiver::vital::algo::triangulate_landmarks::*)
           ( kwiver::vital::camera_map_sptr,
             kwiver::vital::track_map_t,
             kwiver::vital::landmark_map_sptr&
           ) const
         )
         &kwiver::vital::algo::triangulate_landmarks::triangulate);
}
}
}
}
