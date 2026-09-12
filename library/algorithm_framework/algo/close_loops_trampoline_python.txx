// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef CLOSE_LOOPS_TRAMPOLINE_TXX
#define CLOSE_LOOPS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/close_loops.h>

namespace kwiver::vital::python {

template< class close_loops_base = kwiver::vital::algo::close_loops >
class close_loops_trampoline
    : public algorithm_trampoline< close_loops_base >
{
  public:
    using algorithm_trampoline< close_loops_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::feature_track_set_sptr
  stitch(::kwiver::vital::frame_id_t frame_number, ::kwiver::vital::feature_track_set_sptr input, ::kwiver::vital::image_container_sptr image, ::kwiver::vital::image_container_sptr mask) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::feature_track_set_sptr,
      kwiver::vital::algo::close_loops,
      stitch,
      frame_number, input, image, mask
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
