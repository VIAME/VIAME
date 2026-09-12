// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef TRACK_OBJECTS_TRAMPOLINE_TXX
#define TRACK_OBJECTS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/track_objects.h>

namespace kwiver::vital::python {

template< class track_objects_base = kwiver::vital::algo::track_objects >
class track_objects_trampoline
    : public algorithm_trampoline< track_objects_base >
{
  public:
    using algorithm_trampoline< track_objects_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::object_track_set_sptr
  track(::kwiver::vital::timestamp ts, ::kwiver::vital::image_container_sptr image, ::kwiver::vital::detected_object_set_sptr detections) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::object_track_set_sptr,
      kwiver::vital::algo::track_objects,
      track,
      ts, image, detections
      );
  }

  kwiver::vital::object_track_set_sptr
  track(::kwiver::vital::timestamp ts, ::kwiver::vital::image_container_sptr image, ::kwiver::vital::detected_object_set_sptr detections, ::kwiver::vital::f2f_homography_sptr src_to_ref) const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::object_track_set_sptr,
      kwiver::vital::algo::track_objects,
      track,
      ts, image, detections, src_to_ref
      );
  }

  kwiver::vital::object_track_set_sptr
  track(::kwiver::vital::timestamp ts, ::kwiver::vital::image_container_sptr image, ::kwiver::vital::detected_object_set_sptr detections, ::kwiver::vital::object_track_set_sptr existing_tracks) const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::object_track_set_sptr,
      kwiver::vital::algo::track_objects,
      track,
      ts, image, detections, existing_tracks
      );
  }

  kwiver::vital::object_track_set_sptr
  initialize(::kwiver::vital::timestamp ts, ::kwiver::vital::image_container_sptr image, ::kwiver::vital::detected_object_set_sptr seed_detections) const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::object_track_set_sptr,
      kwiver::vital::algo::track_objects,
      initialize,
      ts, image, seed_detections
      );
  }

  kwiver::vital::object_track_set_sptr
  finalize() const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::object_track_set_sptr,
      kwiver::vital::algo::track_objects,
      finalize,
      
      );
  }

  void
  reset() const override
  {
    PYBIND11_OVERLOAD(
      void,
      kwiver::vital::algo::track_objects,
      reset,
      
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
