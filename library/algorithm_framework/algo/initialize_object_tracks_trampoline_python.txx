// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INITIALIZE_OBJECT_TRACKS_TRAMPOLINE_TXX
#define INITIALIZE_OBJECT_TRACKS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/initialize_object_tracks.h>

namespace viame::python {

template< class initialize_object_tracks_base = viame::algo::initialize_object_tracks >
class initialize_object_tracks_trampoline
    : public algorithm_trampoline< initialize_object_tracks_base >
{
  public:
    using algorithm_trampoline< initialize_object_tracks_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::object_track_set_sptr
  initialize(::viame::timestamp ts, ::viame::image_container_sptr image, ::viame::detected_object_set_sptr detections) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::object_track_set_sptr,
      viame::algo::initialize_object_tracks,
      initialize,
      ts, image, detections
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
