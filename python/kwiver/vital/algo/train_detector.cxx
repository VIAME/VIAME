// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/train_detector_trampoline.txx>
#include <python/kwiver/vital/algo/train_detector.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void train_detector(py::module &m)
{
  py::class_< kwiver::vital::algo::train_detector,
              std::shared_ptr<kwiver::vital::algo::train_detector>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::train_detector>,
              train_detector_trampoline<> >(m, "TrainDetector")
    .def(py::init())
    .def_static("static_type_name",
        &kwiver::vital::algo::train_detector::static_type_name)
    .def("train_from_disk",
         &kwiver::vital::algo::train_detector::train_from_disk)
    .def("train_from_memory",
         &kwiver::vital::algo::train_detector::train_from_memory);
}
}
}
}
