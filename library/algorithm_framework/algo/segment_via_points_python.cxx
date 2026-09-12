// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/segment_via_points.h>
#include "algorithm_python.txx"
#include "segment_via_points_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void segment_via_points(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::segment_via_points,
               std::shared_ptr<kwiver::vital::algo::segment_via_points>,
               kwiver::vital::algorithm,
               segment_via_points_trampoline<> > instance(m,  "SegmentViaPoints");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::segment_via_points::interface_name)
    .def("segment", &kwiver::vital::algo::segment_via_points::segment, py::doc(R"( Perform point-based segmentation on an image.

 \param image The image to segment

 \param points Vector of 2D point coordinates [x, y] indicating
        locations for segmentation prompts

 \param point_labels Vector of labels corresponding to each point:
        - 1: foreground (object to segment)
        - 0: background (region to exclude)
        Must have same length as points vector.

 \returns DetectedObjectSet containing segmented objects.
          Each DetectedObject includes:
          - Bounding box around the segmented region
          - Confidence score from the segmentation model
          - Binary mask of the segmented region
)"), py::arg("image"), py::arg("points"), py::arg("point_labels"))
    ;
  register_algorithm< kwiver::vital::algo::segment_via_points > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
