// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/draw_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/draw_tracks.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void draw_tracks(py::module &m)
{
  py::class_< kwiver::vital::algo::draw_tracks,
              std::shared_ptr<kwiver::vital::algo::draw_tracks>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::draw_tracks>,
              draw_tracks_trampoline<> >( m, "DrawTracks" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::draw_tracks::static_type_name)
    .def("draw",
         &kwiver::vital::algo::draw_tracks::draw);
}
}
}
}
