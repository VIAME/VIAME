// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/associate_detections_to_tracks.h>
#include "algorithm_python.txx"
#include "associate_detections_to_tracks_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void associate_detections_to_tracks(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::associate_detections_to_tracks,
               std::shared_ptr<viame::algo::associate_detections_to_tracks>,
               viame::algorithm,
               associate_detections_to_tracks_trampoline<> > instance(m,  "AssociateDetectionsToTracks");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::associate_detections_to_tracks::interface_name)
    .def("associate", &viame::algo::associate_detections_to_tracks::associate, py::doc(R"( Use cost matrices to assign detections to existing tracks

 \param ts frame ID
 \param image contains the input image for the current frame
 \param tracks active track set from the last frame
 \param detections detected object sets from the current frame
 \param matrix matrix containing detection to track association scores
 \param output the output updated detection set
 \param unused output detection set for any detections not associated
 \returns whether or not any tracks were updated)"), py::arg("ts"), py::arg("image"), py::arg("tracks"), py::arg("detections"), py::arg("matrix"), py::arg("output"), py::arg("unused"))
    ;
  register_algorithm< viame::algo::associate_detections_to_tracks > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
