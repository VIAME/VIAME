// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef COMPUTE_REF_HOMOGRAPHY_TRAMPOLINE_TXX
#define COMPUTE_REF_HOMOGRAPHY_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/compute_ref_homography.h>

namespace kwiver::vital::python {

template< class compute_ref_homography_base = kwiver::vital::algo::compute_ref_homography >
class compute_ref_homography_trampoline
    : public algorithm_trampoline< compute_ref_homography_base >
{
  public:
    using algorithm_trampoline< compute_ref_homography_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  kwiver::vital::f2f_homography_sptr
  estimate(::kwiver::vital::frame_id_t frame_number, ::kwiver::vital::feature_track_set_sptr tracks) const override
  {
    PYBIND11_OVERLOAD_PURE(
      kwiver::vital::f2f_homography_sptr,
      kwiver::vital::algo::compute_ref_homography,
      estimate,
      frame_number, tracks
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
