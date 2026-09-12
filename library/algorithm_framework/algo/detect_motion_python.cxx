// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/detect_motion.h>
#include "algorithm_python.txx"
#include "detect_motion_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void detect_motion(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::detect_motion,
               std::shared_ptr<kwiver::vital::algo::detect_motion>,
               kwiver::vital::algorithm,
               detect_motion_trampoline<> > instance(m,  "DetectMotion");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::detect_motion::interface_name)
    .def("process_image", &kwiver::vital::algo::detect_motion::process_image, py::doc(R"( Detect motion from a sequence of images

 This method detects motion of foreground objects within a
 sequence of images in which the background remains stationary.
 Sequential images are passed one at a time. Motion estimates
 are returned for each image as a heat map with higher values
 indicating greater confidence.

 \param ts Timestamp for the input image
 \param image Image from a sequence
 \param reset_model Indicates that the background model should
 be reset, for example, due to changes in lighting condition or
 camera pose

 \returns A heat map image is returned indicating the confidence
 that motion occurred at each pixel. Heat map image is single channel
 and has the same width and height dimensions as the input image.)"), py::arg("ts"), py::arg("image"), py::arg("reset_model"))
    ;
  register_algorithm< kwiver::vital::algo::detect_motion > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
