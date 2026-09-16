// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef RESECTION_CAMERA_TRAMPOLINE_TXX
#define RESECTION_CAMERA_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/resection_camera.h>

namespace viame::python {

template< class resection_camera_base = viame::algo::resection_camera >
class resection_camera_trampoline
    : public algorithm_trampoline< resection_camera_base >
{
  public:
    using algorithm_trampoline< resection_camera_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::camera_perspective_sptr
  resection(::std::vector<viame::vector_<2, double> > const & image_points, ::std::vector<viame::vector_<3, double> > const & world_points, ::viame::camera_intrinsics_sptr initial_calibration, ::std::vector<bool> * inliers) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::camera_perspective_sptr,
      viame::algo::resection_camera,
      resection,
      image_points, world_points, initial_calibration, inliers
      );
  }

  viame::camera_perspective_sptr
  resection(::viame::frame_id_t frame_id, ::viame::landmark_map_sptr landmarks, ::viame::feature_track_set_sptr tracks, ::size_t width, ::size_t height, ::std::unordered_set<long> * inliers) const override
  {
    PYBIND11_OVERLOAD(
      viame::camera_perspective_sptr,
      viame::algo::resection_camera,
      resection,
      frame_id, landmarks, tracks, width, height, inliers
      );
  }

  viame::camera_perspective_sptr
  resection(::viame::frame_id_t frame_id, ::viame::landmark_map_sptr landmarks, ::viame::feature_track_set_sptr tracks, ::viame::camera_intrinsics_sptr initial_calibration, ::std::unordered_set<long> * inliers) const override
  {
    PYBIND11_OVERLOAD(
      viame::camera_perspective_sptr,
      viame::algo::resection_camera,
      resection,
      frame_id, landmarks, tracks, initial_calibration, inliers
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
