// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/filter_tracks.h>
#include <python/kwiver/vital/algo/trampoline/filter_tracks_trampoline.txx>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void filter_tracks(py::module &m)
{
  py::class_< kwiver::vital::algo::filter_tracks,
              std::shared_ptr<kwiver::vital::algo::filter_tracks>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::filter_tracks>,
              filter_tracks_trampoline<> >( m, "FilterTracks" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::filter_tracks::static_type_name)
    .def("filter",
         &kwiver::vital::algo::filter_tracks::filter);
}
}
}
}
