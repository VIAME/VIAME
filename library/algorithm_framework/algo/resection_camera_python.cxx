// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/resection_camera.h>
#include "algorithm_python.txx"
#include "resection_camera_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void resection_camera(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::resection_camera,
               std::shared_ptr<viame::algo::resection_camera>,
               viame::algorithm,
               resection_camera_trampoline<> > instance(m,  "ResectionCamera");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::resection_camera::interface_name)
    .def("resection", (viame::camera_perspective_sptr (viame::algo::resection_camera::*)(::std::vector<viame::vector_<2, double> > const &, ::std::vector<viame::vector_<3, double> > const &, ::viame::camera_intrinsics_sptr, ::std::vector<bool> *) const) &viame::algo::resection_camera::resection, py::doc(R"( Estimate camera parameters from 3D points and their corresponding
 projections.

 \param [in] image_points
   the 2D image space locations which are projections of \p world_points
 \param [in] world_points
   locations in 3D world space corresponding to the \p image_points
 \param [in] initial_calibration
   initial guess on intrinsic parameters of the camera
 \param [out] inliers estimated inlier status for the point pairs
 \return estimated camera parameters)"), py::arg("image_points"), py::arg("world_points"), py::arg("initial_calibration"), py::arg("inliers"))
    .def("resection", (viame::camera_perspective_sptr (viame::algo::resection_camera::*)(::viame::frame_id_t, ::viame::landmark_map_sptr, ::viame::feature_track_set_sptr, ::size_t, ::size_t, ::std::unordered_set<long> *) const) &viame::algo::resection_camera::resection, py::doc(R"( Estimate camera parameters for a frame from landmarks and tracks.

 This is a convenience function for resectioning a camera for a particular
 frame number in a collection of tracks with corresponding landmarks.
 This function extracts corresponding image and worlds points from the
 \p tracks and \p landmarks and then calls resection on those.
 The image \p width and \p height are used to construct an initial
 guess of camera intrinsics.

 \param [in] frame_id frame number for which to estimate a camera
 \param [in] landmarks 3D landmark locations to constrain camera
 \param [in] tracks 2D feature tracks in image coordinates
 \param [in] width image size in the x dimension in pixels
 \param [in] height image size in the y dimension in pixels
 \param [out] inliers landmark identifiers of inliers
 \return estimated camera parameters)"), py::arg("frame_id"), py::arg("landmarks"), py::arg("tracks"), py::arg("width"), py::arg("height"), py::arg("inliers"))
    .def("resection", (viame::camera_perspective_sptr (viame::algo::resection_camera::*)(::viame::frame_id_t, ::viame::landmark_map_sptr, ::viame::feature_track_set_sptr, ::viame::camera_intrinsics_sptr, ::std::unordered_set<long> *) const) &viame::algo::resection_camera::resection, py::doc(R"( Estimate camera parameters for a frame from landmarks and tracks.

 This is a convenience function for resectioning a camera for a particular
 frame number in a collection of tracks with corresponding landmarks.
 This function extracts corresponding image and worlds points from the
 \p tracks and \p landmarks and then calls resection on those.

 \param [in] frame_id frame number for which to estimate a camera
 \param [in] landmarks 3D landmarks locations to constrain camera
 \param [in] tracks 2D feature tracks in image coordinates
 \param [in] initial_calibration
   initial guess on intrinsic parameters of the camera
 \param [out] inliers landmark identifiers of inliers
 \return estimated camera parameters)"), py::arg("frame_id"), py::arg("landmarks"), py::arg("tracks"), py::arg("initial_calibration"), py::arg("inliers"))
    ;
  register_algorithm< viame::algo::resection_camera > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
