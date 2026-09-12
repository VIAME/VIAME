// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/detect_features.h>
#include "algorithm_python.txx"
#include "detect_features_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void detect_features(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::detect_features,
               std::shared_ptr<kwiver::vital::algo::detect_features>,
               kwiver::vital::algorithm,
               detect_features_trampoline<> > instance(m,  "DetectFeatures");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::detect_features::interface_name)
    .def("detect", &kwiver::vital::algo::detect_features::detect, py::doc(R"( Extract a set of image features from the provided image

 A given mask image should be one-channel (mask->depth() == 1). If the
 given mask image has more than one channel, only the first will be
 considered.

 \throws image_size_mismatch_exception
    When the given non-zero mask image does not match the size of the
    dimensions of the given image data.

 \param image_data contains the image data to process
 \param mask Mask image where regions of positive values (boolean true)
             indicate regions to consider. Only the first channel will be
             considered.
 \returns a set of image features)"), py::arg("image_data"), py::arg("mask") = py::none())
    ;
  register_algorithm< kwiver::vital::algo::detect_features > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
