// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/draw_detected_object_set.h>
#include "algorithm_python.txx"
#include "draw_detected_object_set_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void draw_detected_object_set(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::draw_detected_object_set,
               std::shared_ptr<kwiver::vital::algo::draw_detected_object_set>,
               kwiver::vital::algorithm,
               draw_detected_object_set_trampoline<> > instance(m,  "DrawDetectedObjectSet");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::draw_detected_object_set::interface_name)
    .def("draw", &kwiver::vital::algo::draw_detected_object_set::draw, py::doc(R"( Draw detected object boxes on Image.

 This method draws the detections on a copy of the image. The
 input image is unmodified. The actual boxes that are drawn are
 controlled by the configuration for the implementation.

 @param detected_set Set of detected objects
 @param image Boxes are drawn in this image

 @return Image with boxes and other annotations added.)"), py::arg("detected_set"), py::arg("image"))
    ;
  register_algorithm< kwiver::vital::algo::draw_detected_object_set > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
