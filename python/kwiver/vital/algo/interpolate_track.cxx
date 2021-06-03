// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/interpolate_track_trampoline.txx>
#include <python/kwiver/vital/algo/interpolate_track.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void interpolate_track(py::module &m)
{
  py::class_< kwiver::vital::algo::interpolate_track,
              std::shared_ptr<kwiver::vital::algo::interpolate_track>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::interpolate_track>,
              interpolate_track_trampoline<> >(m, "InterpolateTrack")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::interpolate_track::static_type_name)
    .def("interpolate",
        &kwiver::vital::algo::interpolate_track::interpolate)
    .def("set_video_input",
        &kwiver::vital::algo::interpolate_track::set_video_input)
    .def("set_progress_callback",
        &kwiver::vital::algo::interpolate_track::set_progress_callback);
}
}
}
}
