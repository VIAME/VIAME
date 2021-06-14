// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/match_features_trampoline.txx>
#include <python/kwiver/vital/algo/match_features.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void match_features(py::module &m)
{
  py::class_< kwiver::vital::algo::match_features,
              std::shared_ptr<kwiver::vital::algo::match_features>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::match_features>,
              match_features_trampoline<> >(m, "MatchFeatures")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::match_features::static_type_name)
    .def("match",
        &kwiver::vital::algo::match_features::match);
}
}
}
}
