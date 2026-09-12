// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/compute_ref_homography.h>
#include "algorithm_python.txx"
#include "compute_ref_homography_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void compute_ref_homography(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::compute_ref_homography,
               std::shared_ptr<kwiver::vital::algo::compute_ref_homography>,
               kwiver::vital::algorithm,
               compute_ref_homography_trampoline<> > instance(m,  "ComputeRefHomography");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::compute_ref_homography::interface_name)
    .def("estimate", &kwiver::vital::algo::compute_ref_homography::estimate, py::doc(R"( Estimate the transformation which maps some frame to a reference frame

 Similarly to track_features, this class was designed to be called in
 an online fashion for each sequential frame. The output homography
 will contain a transformation mapping points from the current frame
 (with frame_id frame_number) to the earliest possible reference frame
 via post multiplying points on the current frame with the computed
 homography.

 The returned homography is internally allocated and passed back
 through a smart pointer transferring ownership of the memory to
 the caller.

 \param [in]   frame_number frame identifier for the current frame
 \param [in]   tracks the set of all tracked features from the image
 \return estimated homography)"), py::arg("frame_number"), py::arg("tracks"))
    ;
  register_algorithm< kwiver::vital::algo::compute_ref_homography > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
