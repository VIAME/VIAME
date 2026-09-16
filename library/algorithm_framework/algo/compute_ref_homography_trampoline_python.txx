// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef COMPUTE_REF_HOMOGRAPHY_TRAMPOLINE_TXX
#define COMPUTE_REF_HOMOGRAPHY_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/compute_ref_homography.h>

namespace viame::python {

template< class compute_ref_homography_base = viame::algo::compute_ref_homography >
class compute_ref_homography_trampoline
    : public algorithm_trampoline< compute_ref_homography_base >
{
  public:
    using algorithm_trampoline< compute_ref_homography_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  viame::f2f_homography_sptr
  estimate(::viame::frame_id_t frame_number, ::viame::feature_track_set_sptr tracks) const override
  {
    PYBIND11_OVERLOAD_PURE(
      viame::f2f_homography_sptr,
      viame::algo::compute_ref_homography,
      estimate,
      frame_number, tracks
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
