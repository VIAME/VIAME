// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/detect_motion_trampoline.txx>
#include <python/kwiver/vital/algo/detect_motion.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void detect_motion(py::module &m)
{
  py::class_< kwiver::vital::algo::detect_motion,
              std::shared_ptr<kwiver::vital::algo::detect_motion>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::detect_motion>,
              detect_motion_trampoline<> >( m, "DetectMotion" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::detect_motion::static_type_name)
    .def("process_image",
         &kwiver::vital::algo::detect_motion::process_image);
}
}
}
}
