// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/image_object_detector.h>
#include "algorithm_python.txx"
#include "image_object_detector_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void image_object_detector(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::image_object_detector,
               std::shared_ptr<kwiver::vital::algo::image_object_detector>,
               kwiver::vital::algorithm,
               image_object_detector_trampoline<> > instance(m,  "ImageObjectDetector");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::image_object_detector::interface_name)
    .def("detect", &kwiver::vital::algo::image_object_detector::detect, py::doc(R"( Find all objects on the provided image

 This method analyzes the supplied image and along with any saved
 context, returns a vector of detected image objects.

 \param image_data the image pixels
 \returns vector of image objects found)"), py::arg("image_data"))
    .def("batch_detect", &kwiver::vital::algo::image_object_detector::batch_detect, py::doc(R"( Detect objects in a batch of images

 This method processes multiple images at once, which can be more
 efficient for some detectors (e.g., those using batch processing on GPU).

 The default implementation simply calls detect() on each image.

 \param images Vector of images to process
 \returns Vector of detection sets, one per input image)"), py::arg("images"))
    ;
  register_algorithm< kwiver::vital::algo::image_object_detector > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
