// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef REFINE_TRACKS_TRAMPOLINE_TXX
#define REFINE_TRACKS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/refine_tracks.h>

namespace viame::python {

template< class refine_tracks_base = viame::algo::refine_tracks >
class refine_tracks_trampoline
    : public algorithm_trampoline< refine_tracks_base >
{
  public:
    using algorithm_trampoline< refine_tracks_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::object_track_set_sptr
  refine(::viame::timestamp ts, ::viame::image_container_sptr image_data, ::viame::object_track_set_sptr tracks) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::object_track_set_sptr,
      viame::algo::refine_tracks,
      refine,
      ts, image_data, tracks
      );
  }

  viame::object_track_set_sptr
  finalize() const override
  {
    PYBIND11_OVERLOAD(
      viame::object_track_set_sptr,
      viame::algo::refine_tracks,
      finalize,
      
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
