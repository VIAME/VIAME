// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef TRACK_OBJECTS_TRAMPOLINE_TXX
#define TRACK_OBJECTS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/track_objects.h>

namespace viame::python {

template< class track_objects_base = viame::algo::track_objects >
class track_objects_trampoline
    : public algorithm_trampoline< track_objects_base >
{
  public:
    using algorithm_trampoline< track_objects_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::object_track_set_sptr
  track(::viame::timestamp ts, ::viame::image_container_sptr image, ::viame::detected_object_set_sptr detections) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::object_track_set_sptr,
      viame::algo::track_objects,
      track,
      ts, image, detections
      );
  }

  viame::object_track_set_sptr
  track(::viame::timestamp ts, ::viame::image_container_sptr image, ::viame::detected_object_set_sptr detections, ::viame::f2f_homography_sptr src_to_ref) const override
  {
    PYBIND11_OVERLOAD(
      viame::object_track_set_sptr,
      viame::algo::track_objects,
      track,
      ts, image, detections, src_to_ref
      );
  }

  viame::object_track_set_sptr
  track(::viame::timestamp ts, ::viame::image_container_sptr image, ::viame::detected_object_set_sptr detections, ::viame::object_track_set_sptr existing_tracks) const override
  {
    PYBIND11_OVERLOAD(
      viame::object_track_set_sptr,
      viame::algo::track_objects,
      track,
      ts, image, detections, existing_tracks
      );
  }

  viame::object_track_set_sptr
  initialize(::viame::timestamp ts, ::viame::image_container_sptr image, ::viame::detected_object_set_sptr seed_detections) const override
  {
    PYBIND11_OVERLOAD(
      viame::object_track_set_sptr,
      viame::algo::track_objects,
      initialize,
      ts, image, seed_detections
      );
  }

  viame::object_track_set_sptr
  finalize() const override
  {
    PYBIND11_OVERLOAD(
      viame::object_track_set_sptr,
      viame::algo::track_objects,
      finalize,
      
      );
  }

  void
  reset() const override
  {
    PYBIND11_OVERLOAD(
      void,
      viame::algo::track_objects,
      reset,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
