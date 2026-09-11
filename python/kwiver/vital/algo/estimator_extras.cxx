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
#include <viame/algorithm_framework/algo/optimize_cameras.h>

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

// ----------------------------------------------------------------------------
/// Give `optimize_cameras`'s two overloads two python names.
///
/// The generated binding exposes both as `optimize`, and python has one name
/// for the two, so a caller reaching for the single-camera form gets the map
/// form's argument check. The map form keeps `optimize` and returns the
/// optimised map rather than writing into the argument -- the interface
/// returns void, so from python it would otherwise return nothing at all --
/// and the single-camera form becomes `optimize_camera`, which is the name
/// the hand-written trampoline looks for on the implementing side.
void
optimize_cameras_extras( py::module& m )
{
  using algorithm = kwiver::vital::algo::optimize_cameras;

  py::object cls = m.attr( "OptimizeCameras" );

  cls.attr( "optimize" ) = py::cpp_function(
    []( algorithm const& self,
        kwiver::vital::camera_map_sptr cameras,
        kwiver::vital::feature_track_set_sptr tracks,
        kwiver::vital::landmark_map_sptr landmarks,
        kwiver::vital::sfm_constraints_sptr constraints )
    {
      self.optimize( cameras, tracks, landmarks, constraints );
      return cameras;
    },
    py::is_method( cls ),
    py::doc( "Optimize a map of cameras. Returns the optimised map; the "
             "C++ signature takes it as an in/out parameter, which python "
             "cannot see." ),
    py::arg( "cameras" ),
    py::arg( "tracks" ),
    py::arg( "landmarks" ),
    py::arg( "constraints" ) = py::none() );

  cls.attr( "optimize_camera" ) = py::cpp_function(
    []( algorithm const& self,
        kwiver::vital::camera_perspective_sptr camera,
        std::vector< kwiver::vital::feature_sptr > const& features,
        std::vector< kwiver::vital::landmark_sptr > const& landmarks,
        kwiver::vital::sfm_constraints_sptr constraints )
    {
      self.optimize( camera, features, landmarks, constraints );
      return camera;
    },
    py::is_method( cls ),
    py::doc( "Optimize one camera against parallel feature and landmark "
             "vectors. Returns the optimised camera." ),
    py::arg( "camera" ),
    py::arg( "features" ),
    py::arg( "landmarks" ),
    py::arg( "constraints" ) = py::none() );
}

} // namespace kwiver::vital::python
