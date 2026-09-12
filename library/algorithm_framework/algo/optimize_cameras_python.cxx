// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/optimize_cameras.h>
#include "algorithm_python.txx"
#include "optimize_cameras_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void optimize_cameras(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::optimize_cameras,
               std::shared_ptr<kwiver::vital::algo::optimize_cameras>,
               kwiver::vital::algorithm,
               optimize_cameras_trampoline<> > instance(m,  "OptimizeCameras");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::optimize_cameras::interface_name)
    .def("optimize", (void (kwiver::vital::algo::optimize_cameras::*)(::kwiver::vital::camera_map_sptr &, ::kwiver::vital::feature_track_set_sptr, ::kwiver::vital::landmark_map_sptr, ::kwiver::vital::sfm_constraints_sptr) const) &kwiver::vital::algo::optimize_cameras::optimize, py::doc(R"( Optimize camera parameters given sets of landmarks and feature tracks

 We only optimize cameras that have associating tracks and landmarks in
 the given maps.  The default implementation collects the corresponding
 features and landmarks for each camera and calls the single camera
 optimize function.

 \throws invalid_value When one or more of the given pointer is Null.

 \param[in,out] cameras   Cameras to optimize.
 \param[in]     tracks    The feature tracks to use as constraints.
 \param[in]     landmarks The landmarks the cameras are viewing.
 \param[in]     metadata  The optional metadata to constrain the
                          optimization.)"), py::arg("cameras"), py::arg("tracks"), py::arg("landmarks"), py::arg("constraints"))
    .def("optimize", (void (kwiver::vital::algo::optimize_cameras::*)(::kwiver::vital::camera_perspective_sptr &, ::std::vector<std::shared_ptr<kwiver::vital::feature> > const &, ::std::vector<std::shared_ptr<kwiver::vital::landmark> > const &, ::kwiver::vital::sfm_constraints_sptr) const) &kwiver::vital::algo::optimize_cameras::optimize, py::doc(R"( Optimize a single camera given corresponding features and landmarks

 This function assumes that 2D features viewed by this camera have
 already been put into correspondence with 3D landmarks by aligning
 them into two parallel vectors

 \param[in,out] camera    The camera to optimize.
 \param[in]     features  The vector of features observed by \p camera
                          to use as constraints.
 \param[in]     landmarks The vector of landmarks corresponding to
                          \p features.
 \param[in]     metadata  The optional metadata to constrain the
                          optimization.)"), py::arg("camera"), py::arg("features"), py::arg("landmarks"), py::arg("constraints"))
    ;
  register_algorithm< kwiver::vital::algo::optimize_cameras > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
