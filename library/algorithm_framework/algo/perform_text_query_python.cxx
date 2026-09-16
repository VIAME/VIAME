// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/perform_text_query.h>
#include "algorithm_python.txx"
#include "perform_text_query_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void perform_text_query(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::perform_text_query,
               std::shared_ptr<viame::algo::perform_text_query>,
               viame::algorithm,
               perform_text_query_trampoline<> > instance(m,  "PerformTextQuery");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::perform_text_query::interface_name)
    .def("perform_query", &viame::algo::perform_text_query::perform_query, py::doc(R"( Perform text-based detection/segmentation on images.

 \param text_query Natural language description of objects to detect.
        Examples: "fish", "red car", "person wearing hat"
        Can be comma-separated for multiple classes: "fish, crab, starfish"

 \param images Vector of images to process. Interpretation depends on
 context:
        - Multiple cameras at same time
        - Multiple frames from video
        - Mixed (e.g., stereo pairs over time)

 \param timestamps Optional timestamps corresponding to each image.
        If provided, must match length of images.
        Enables temporal reasoning and proper track state assignment.
        If empty, images are treated as independent.

 \param input_tracks Optional existing tracks to refine, one per image.
        If provided, must match length of images.
        Detections are associated with existing tracks via IoU matching.
        If empty, new track sets are created from detections.

 \returns Vector of track sets, one per input image.
          Each contains object_track_state entries with bounding box,
          confidence score, classification, and optional polygon mask.
)"), py::arg("text_query"), py::arg("images"), py::arg("timestamps"), py::arg("input_tracks"))
    ;
  register_algorithm< viame::algo::perform_text_query > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
