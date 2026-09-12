// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/initialize_object_tracks.h>
#include "algorithm_python.txx"
#include "initialize_object_tracks_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void initialize_object_tracks(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::initialize_object_tracks,
               std::shared_ptr<kwiver::vital::algo::initialize_object_tracks>,
               kwiver::vital::algorithm,
               initialize_object_tracks_trampoline<> > instance(m,  "InitializeObjectTracks");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::initialize_object_tracks::interface_name)
    .def("initialize", &kwiver::vital::algo::initialize_object_tracks::initialize, py::doc(R"( Initialize new object tracks given detections.

 \param ts frame ID
 \param image contains the input image for the current frame
 \param detections detected object sets from the current frame
 \returns newly initialized tracks)"), py::arg("ts"), py::arg("image"), py::arg("detections"))
    ;
  register_algorithm< kwiver::vital::algo::initialize_object_tracks > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
