// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/initialize_object_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/initialize_object_tracks.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void initialize_object_tracks(py::module &m)
{
  py::class_< kwiver::vital::algo::initialize_object_tracks,
              std::shared_ptr<kwiver::vital::algo::initialize_object_tracks>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::initialize_object_tracks>,
              initialize_object_tracks_trampoline<> >(m, "InitializeObjectTracks")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::initialize_object_tracks::static_type_name)
    .def("initialize",
        &kwiver::vital::algo::initialize_object_tracks::initialize);
}
}
}
}
