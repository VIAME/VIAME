// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/draw_detected_object_set_trampoline.txx>
#include <python/kwiver/vital/algo/draw_detected_object_set.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
void draw_detected_object_set(py::module &m)
{
  py::class_< kwiver::vital::algo::draw_detected_object_set,
              std::shared_ptr<kwiver::vital::algo::draw_detected_object_set>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::draw_detected_object_set>,
              draw_detected_object_set_trampoline<> >( m, "DrawDetectedObjectSet" )
    .def(py::init())
    .def_static("static_type_name",
                &kwiver::vital::algo::draw_detected_object_set::static_type_name)
    .def("draw",
         &kwiver::vital::algo::draw_detected_object_set::draw);
}
}
}
}
