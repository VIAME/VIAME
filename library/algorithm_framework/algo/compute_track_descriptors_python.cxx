// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/compute_track_descriptors.h>
#include "algorithm_python.txx"
#include "compute_track_descriptors_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void compute_track_descriptors(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<viame::algo::compute_track_descriptors,
               std::shared_ptr<viame::algo::compute_track_descriptors>,
               viame::algorithm,
               compute_track_descriptors_trampoline<> > instance(m,  "ComputeTrackDescriptors");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::compute_track_descriptors::interface_name)
    .def("compute", &viame::algo::compute_track_descriptors::compute, py::doc(R"( Compute track descriptors given an image and tracks

 \param ts timestamp for the current frame
 \param image_data contains the image data to process
 \param tracks the tracks to extract descriptors around

 \returns a set of track descriptors)"), py::arg("ts"), py::arg("image_data"), py::arg("tracks"))
    .def("flush", &viame::algo::compute_track_descriptors::flush, py::doc(R"( Flush any remaining in-progress descriptors

 This is typically called at the end of a video, in case
 any temporal descriptors and currently in progress and
 still need to be output.

 \returns a set of track descriptors)"))
    ;
  register_algorithm< viame::algo::compute_track_descriptors > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
