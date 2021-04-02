// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/refine_detections_trampoline.txx>
#include <python/kwiver/vital/algo/refine_detections.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void refine_detections(py::module &m)
{
  py::class_< kwiver::vital::algo::refine_detections,
              std::shared_ptr<kwiver::vital::algo::refine_detections>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::refine_detections>,
              refine_detections_trampoline<> >(m, "RefineDetections")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::refine_detections::static_type_name)
    .def("refine", &kwiver::vital::algo::refine_detections::refine);
}
}
}
}
