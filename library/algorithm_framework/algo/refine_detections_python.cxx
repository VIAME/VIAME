// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/refine_detections.h>
#include "algorithm_python.txx"
#include "refine_detections_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void refine_detections(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::refine_detections,
               std::shared_ptr<viame::algo::refine_detections>,
               viame::algorithm,
               refine_detections_trampoline<> > instance(m,  "RefineDetections");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::refine_detections::interface_name)
    .def("refine", &viame::algo::refine_detections::refine, py::doc(R"( Refine all object detections on the provided image

 This method analyzes the supplied image and and detections on it,
 returning a refined set of detections.

 \param image_data the image pixels
 \param detections detected objects
 \returns vector of image objects refined)"), py::arg("image_data"), py::arg("detections"))
    ;
  register_algorithm< viame::algo::refine_detections > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
