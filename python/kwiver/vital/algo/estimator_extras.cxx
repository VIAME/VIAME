// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

#include <python/kwiver/vital/algo/estimator_extras.h>

// The numpy caster for `vital::vector_2d`. Without it a point list has to be
// built out of the bound C++ type rather than out of arrays, which is not
// what any caller has.
#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>

#include <viame/algorithm_framework/algo/estimate_fundamental_matrix.h>
#include <viame/algorithm_framework/algo/estimate_homography.h>

#include <pybind11/stl.h>

#include <utility>
#include <vector>

namespace kwiver::vital::python {

namespace py = pybind11;

namespace {

// ----------------------------------------------------------------------------
/// Replace an estimator's two `estimate` overloads with ones that return the
/// inliers.
///
/// The generated binding exposes `estimate` exactly as the C++ signature
/// reads, and that signature has `std::vector<bool>& inliers` as a pure
/// output. pybind11 converts a python list to a fresh vector, so the C++
/// fills a temporary and the caller's list is never touched: from python the
/// inliers are simply unobtainable. That is not a detail -- every C++ caller
/// of these estimators uses the inliers rather than the matrix alone
/// (`match_features_homography` keeps only the inlier matches), so a python
/// caller that cannot see them cannot do what the C++ ones do.
///
/// The argument goes away with it, being an output and nothing else, so the
/// python signature is `estimate(pts1, pts2, inlier_scale=1.0)` returning
/// `(matrix, inliers)`. The trampoline in `trampolines/` expects a python
/// *implementation* to have the same shape, so the two sides read alike.
template < typename Algorithm, typename Matrix >
void
bind_estimator( py::module& m, char const* python_name )
{
  py::object cls = m.attr( python_name );

  cls.attr( "estimate" ) = py::cpp_function(
    []( Algorithm const& self,
        std::vector< kwiver::vital::vector_2d > const& pts1,
        std::vector< kwiver::vital::vector_2d > const& pts2,
        double inlier_scale )
    {
      std::vector< bool > inliers;
      auto matrix = self.estimate( pts1, pts2, inliers, inlier_scale );
      return std::make_pair( std::move( matrix ), std::move( inliers ) );
    },
    py::is_method( cls ),
    py::doc( "Estimate from corresponding points. Returns (matrix, inliers); "
             "inliers has one flag per point pair." ),
    py::arg( "pts1" ),
    py::arg( "pts2" ),
    py::arg( "inlier_scale" ) = 1.0 );

  // The overload taking feature sets and a match set. `cpp_function` with
  // `py::is_method` replaces rather than appends, so both are re-bound here
  // as one overload set.
  cls.attr( "estimate_matches" ) = py::cpp_function(
    []( Algorithm const& self,
        kwiver::vital::feature_set_sptr feat1,
        kwiver::vital::feature_set_sptr feat2,
        kwiver::vital::match_set_sptr matches,
        double inlier_scale )
    {
      std::vector< bool > inliers;
      auto matrix =
        self.estimate( feat1, feat2, matches, inliers, inlier_scale );
      return std::make_pair( std::move( matrix ), std::move( inliers ) );
    },
    py::is_method( cls ),
    py::doc( "Estimate from corresponding features. Returns (matrix, "
             "inliers); inliers has one flag per match." ),
    py::arg( "feat1" ),
    py::arg( "feat2" ),
    py::arg( "matches" ),
    py::arg( "inlier_scale" ) = 1.0 );
}

} // namespace

// ----------------------------------------------------------------------------
void
estimator_extras( py::module& m )
{
  bind_estimator< kwiver::vital::algo::estimate_homography,
                  kwiver::vital::homography_sptr >( m, "EstimateHomography" );
  bind_estimator< kwiver::vital::algo::estimate_fundamental_matrix,
                  kwiver::vital::fundamental_matrix_sptr >(
    m, "EstimateFundamentalMatrix" );
}

} // namespace kwiver::vital::python
