// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Hand-written trampoline for estimate_homography
///
/// Replaces the generated one, which cannot carry the `inliers` out
/// parameter back from python. A python implementation is called as
/// `estimate(..., inlier_scale)` and returns `(matrix, inliers)`, which is
/// the same shape `estimator_extras.cxx` gives a python *caller*. See
/// trampolines/README.md.

#ifndef ESTIMATE_HOMOGRAPHY_TRAMPOLINE_TXX
#define ESTIMATE_HOMOGRAPHY_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <python/kwiver/vital/algo/algorithm_trampoline.txx>
#include <python/kwiver/vital/algo/out_parameter.txx>
#include <viame/algorithm_framework/algo/estimate_homography.h>

namespace kwiver::vital::python {

template< class estimate_homography_base =
            kwiver::vital::algo::estimate_homography >
class estimate_homography_trampoline
  : public algorithm_trampoline< estimate_homography_base >
{
public:
  using algorithm_trampoline< estimate_homography_base >::algorithm_trampoline;

  kwiver::vital::homography_sptr
  estimate(
    ::kwiver::vital::feature_set_sptr feat1,
    ::kwiver::vital::feature_set_sptr feat2,
    ::kwiver::vital::match_set_sptr matches,
    ::std::vector< bool >& inliers,
    double inlier_scale ) const override
  {
    // The two C++ overloads share one python name, so a python
    // implementation that defines `estimate` means the *points* overload:
    // it is the pure one, and the only one an implementation has to write.
    // This overload has a C++ body that turns features and matches into
    // points and calls that one, which is what should run unless python
    // has deliberately overridden it -- under a name of its own, the same
    // `estimate_matches` a python caller sees.
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< kwiver::vital::algo::estimate_homography const* >( this ),
        "estimate_matches" );

    if( !overload )
    {
      return estimate_homography_base::estimate(
        feat1, feat2, matches, inliers, inlier_scale );
    }

    return unpack_out_parameters< kwiver::vital::homography_sptr >(
      overload( feat1, feat2, matches, inlier_scale ),
      "estimate_homography.estimate_matches", inliers );
  }

  kwiver::vital::homography_sptr
  estimate(
    ::std::vector< kwiver::vital::vector_< 2, double > > const& pts1,
    ::std::vector< kwiver::vital::vector_< 2, double > > const& pts2,
    ::std::vector< bool >& inliers,
    double inlier_scale ) const override
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< kwiver::vital::algo::estimate_homography const* >( this ),
        "estimate" );

    if( !overload )
    {
      pybind11::pybind11_fail(
        "Tried to call pure virtual function "
        "\"estimate_homography::estimate\"" );
    }

    return unpack_out_parameters< kwiver::vital::homography_sptr >(
      overload( pts1, pts2, inlier_scale ),
      "estimate_homography.estimate", inliers );
  }
};

} // namespace kwiver::vital::python

#undef KWIVER_PYBIND11_INCLUDE
#endif
