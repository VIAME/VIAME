// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/associate_detections_to_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/associate_detections_to_tracks.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void associate_detections_to_tracks(py::module &m)
{
  py::class_< kwiver::vital::algo::associate_detections_to_tracks,
              std::shared_ptr<kwiver::vital::algo::associate_detections_to_tracks>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::associate_detections_to_tracks>,
              associate_detections_to_tracks_trampoline<> >( m,
                                                 "AssociateDetectionsToTracks" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::associate_detections_to_tracks::static_type_name)
    .def("associate",
         &kwiver::vital::algo::associate_detections_to_tracks::associate);
}
}
}
}
