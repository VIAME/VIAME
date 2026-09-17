// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/train_detector.h>
#include "algorithm_python.txx"
#include "train_detector_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void train_detector(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::train_detector,
               std::shared_ptr<viame::algo::train_detector>,
               viame::algorithm,
               train_detector_trampoline<> > instance(m,  "TrainDetector");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::train_detector::interface_name)
    .def("add_data_from_disk", &viame::algo::train_detector::add_data_from_disk, py::doc(R"( Add training data from disk

 This varient is geared towards offline training.

 \param object_labels object category labels for training
 \param train_image_list list of train image filenames
 \param train_groundtruth annotations loaded for each image
 \param test_image_list list of test image filenames
 \param test_groundtruth annotations loaded for each image)"), py::arg("object_labels"), py::arg("train_image_names"), py::arg("train_groundtruth"), py::arg("test_image_names"), py::arg("test_groundtruth"))
    .def("add_data_from_memory", &viame::algo::train_detector::add_data_from_memory, py::doc(R"( Add training data from memory

 This varient is geared towards online training, and is not required
 to be defined.

 \throws runtime_exception if not defined.

 \param object_labels object category labels for training
 \param train_images vector of input train images
 \param train_groundtruth annotations loaded for each train image
 \param test_images optional vector of input test images
 \param test_groundtruth optional annotations loaded for each test image)"), py::arg("object_labels"), py::arg("train_images"), py::arg("train_groundtruth"), py::arg("test_images"), py::arg("test_groundtruth"))
    .def("update_model", &viame::algo::train_detector::update_model, py::doc(R"( Train a detection model given all loaded data

 This varient is geared towards either offline or online training
 depending on the implementation.

 \throws runtime_exception if not defined or there's a data issue.

 \returns Map containing locations of final model files or other
          general configuration parameters for model inference.)"))
    ;
  register_algorithm< viame::algo::train_detector > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
