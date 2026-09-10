// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Hand-written trampoline for estimate_fundamental_matrix
///
/// Replaces the generated one, which cannot carry the `inliers` out
/// parameter back from python. A python implementation is called as
/// `estimate(..., inlier_scale)` and returns `(matrix, inliers)`, which is
/// the same shape `estimator_extras.cxx` gives a python *caller*. See
/// trampolines/README.md.

#ifndef ESTIMATE_FUNDAMENTAL_MATRIX_TRAMPOLINE_TXX
#define ESTIMATE_FUNDAMENTAL_MATRIX_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <python/kwiver/vital/algo/algorithm_trampoline.txx>
#include <python/kwiver/vital/algo/out_parameter.txx>
#include <viame/algorithm_framework/algo/estimate_fundamental_matrix.h>

namespace kwiver::vital::python {

template< class estimate_fundamental_matrix_base =
            kwiver::vital::algo::estimate_fundamental_matrix >
class estimate_fundamental_matrix_trampoline
  : public algorithm_trampoline< estimate_fundamental_matrix_base >
{
public:
  using algorithm_trampoline< estimate_fundamental_matrix_base >::algorithm_trampoline;

  kwiver::vital::fundamental_matrix_sptr
  estimate(
    ::kwiver::vital::feature_set_sptr const feat1,
    ::kwiver::vital::feature_set_sptr const feat2,
    ::kwiver::vital::match_set_sptr const matches,
    ::std::vector< bool >& inliers,
    double inlier_scale ) const override
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< kwiver::vital::algo::estimate_fundamental_matrix const* >( this ),
        "estimate" );

    if( !overload )
    {
      return estimate_fundamental_matrix_base::estimate(
        feat1, feat2, matches, inliers, inlier_scale );
    }

    return unpack_out_parameters< kwiver::vital::fundamental_matrix_sptr >(
      overload( feat1, feat2, matches, inlier_scale ),
      "estimate_fundamental_matrix.estimate", inliers );
  }

  kwiver::vital::fundamental_matrix_sptr
  estimate(
    ::std::vector< kwiver::vital::vector_< 2, double > > const& pts1,
    ::std::vector< kwiver::vital::vector_< 2, double > > const& pts2,
    ::std::vector< bool >& inliers,
    double inlier_scale ) const override
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< kwiver::vital::algo::estimate_fundamental_matrix const* >( this ),
        "estimate" );

    if( !overload )
    {
      pybind11::pybind11_fail(
        "Tried to call pure virtual function "
        "\"estimate_fundamental_matrix::estimate\"" );
    }

    return unpack_out_parameters< kwiver::vital::fundamental_matrix_sptr >(
      overload( pts1, pts2, inlier_scale ),
      "estimate_fundamental_matrix.estimate", inliers );
  }
};

} // namespace kwiver::vital::python

#undef KWIVER_PYBIND11_INCLUDE
#endif
