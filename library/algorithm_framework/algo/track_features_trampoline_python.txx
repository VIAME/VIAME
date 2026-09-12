// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef TRACK_FEATURES_TRAMPOLINE_TXX
#define TRACK_FEATURES_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/track_features.h>

namespace kwiver::vital::python {

template< class track_features_base = kwiver::vital::algo::track_features >
class track_features_trampoline
    : public algorithm_trampoline< track_features_base >
{
  public:
    using algorithm_trampoline< track_features_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::feature_track_set_sptr
  track(::kwiver::vital::feature_track_set_sptr prev_tracks, ::kwiver::vital::frame_id_t frame_number, ::kwiver::vital::image_container_sptr image_data, ::kwiver::vital::image_container_sptr mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::feature_track_set_sptr,
      kwiver::vital::algo::track_features,
      track,
      prev_tracks, frame_number, image_data, mask
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
