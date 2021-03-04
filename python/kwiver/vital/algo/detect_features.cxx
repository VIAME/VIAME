// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/detect_features_trampoline.txx>
#include <python/kwiver/vital/algo/detect_features.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void detect_features(py::module &m)
{
  py::class_< kwiver::vital::algo::detect_features,
              std::shared_ptr<kwiver::vital::algo::detect_features>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::detect_features>,
              detect_features_trampoline<> >( m, "DetectFeatures" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::detect_features::static_type_name)
    .def("detect",
         &kwiver::vital::algo::detect_features::detect);
}
}
}
}
