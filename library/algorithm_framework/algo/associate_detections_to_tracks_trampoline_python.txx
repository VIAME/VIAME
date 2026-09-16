// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ASSOCIATE_DETECTIONS_TO_TRACKS_TRAMPOLINE_TXX
#define ASSOCIATE_DETECTIONS_TO_TRACKS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/associate_detections_to_tracks.h>

namespace viame::python {

template< class associate_detections_to_tracks_base = viame::algo::associate_detections_to_tracks >
class associate_detections_to_tracks_trampoline
    : public algorithm_trampoline< associate_detections_to_tracks_base >
{
  public:
    using algorithm_trampoline< associate_detections_to_tracks_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bool
  associate(::viame::timestamp ts, ::viame::image_container_sptr image, ::viame::object_track_set_sptr tracks, ::viame::detected_object_set_sptr detections, ::viame::matrix_d matrix, ::viame::object_track_set_sptr & output, ::viame::detected_object_set_sptr & unused) const override
  {
    PYBIND11_OVERLOAD_PURE(
      bool,
      viame::algo::associate_detections_to_tracks,
      associate,
      ts, image, tracks, detections, matrix, output, unused
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
