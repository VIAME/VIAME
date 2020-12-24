// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/image_object_detector_trampoline.txx>
#include <python/kwiver/vital/algo/image_object_detector.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void image_object_detector(py::module &m)
{
  py::class_< kwiver::vital::algo::image_object_detector,
              std::shared_ptr<kwiver::vital::algo::image_object_detector>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::image_object_detector>,
              image_object_detector_trampoline<> >(m, "ImageObjectDetector")
    .def(py::init())
    .def_static("static_type_name", &kwiver::vital::algo::image_object_detector::static_type_name)
    .def("detect", &kwiver::vital::algo::image_object_detector::detect);
}
}
}
}
