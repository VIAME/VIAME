// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef TRACK_FEATURES_TRAMPOLINE_TXX
#define TRACK_FEATURES_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/track_features.h>

namespace viame::python {

template< class track_features_base = viame::algo::track_features >
class track_features_trampoline
    : public algorithm_trampoline< track_features_base >
{
  public:
    using algorithm_trampoline< track_features_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::feature_track_set_sptr
  track(::viame::feature_track_set_sptr prev_tracks, ::viame::frame_id_t frame_number, ::viame::image_container_sptr image_data, ::viame::image_container_sptr mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::feature_track_set_sptr,
      viame::algo::track_features,
      track,
      prev_tracks, frame_number, image_data, mask
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
