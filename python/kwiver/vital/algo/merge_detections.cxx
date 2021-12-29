// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <python/kwiver/vital/algo/trampoline/merge_detections_trampoline.txx>
#include <python/kwiver/vital/algo/merge_detections.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void merge_detections(py::module &m)
{
  py::class_< kwiver::vital::algo::merge_detections,
              std::shared_ptr<kwiver::vital::algo::merge_detections>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::merge_detections>,
              merge_detections_trampoline<> >(m, "MergeDetections")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::merge_detections::static_type_name)
    .def("merge",
        &kwiver::vital::algo::merge_detections::merge);
}
}
}
}
