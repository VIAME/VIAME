// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INITIALIZE_OBJECT_TRACKS_TRAMPOLINE_TXX
#define INITIALIZE_OBJECT_TRACKS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/initialize_object_tracks.h>

namespace kwiver::vital::python {

template< class initialize_object_tracks_base = kwiver::vital::algo::initialize_object_tracks >
class initialize_object_tracks_trampoline
    : public algorithm_trampoline< initialize_object_tracks_base >
{
  public:
    using algorithm_trampoline< initialize_object_tracks_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::object_track_set_sptr
  initialize(::kwiver::vital::timestamp ts, ::kwiver::vital::image_container_sptr image, ::kwiver::vital::detected_object_set_sptr detections) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::object_track_set_sptr,
      kwiver::vital::algo::initialize_object_tracks,
      initialize,
      ts, image, detections
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
