// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef CLOSE_LOOPS_TRAMPOLINE_TXX
#define CLOSE_LOOPS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/close_loops.h>

namespace viame::python {

template< class close_loops_base = viame::algo::close_loops >
class close_loops_trampoline
    : public algorithm_trampoline< close_loops_base >
{
  public:
    using algorithm_trampoline< close_loops_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::feature_track_set_sptr
  stitch(::viame::frame_id_t frame_number, ::viame::feature_track_set_sptr input, ::viame::image_container_sptr image, ::viame::image_container_sptr mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::feature_track_set_sptr,
      viame::algo::close_loops,
      stitch,
      frame_number, input, image, mask
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
