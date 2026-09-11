// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Hand-written trampoline for optimize_cameras
///
/// Replaces the generated one, which cannot carry the optimised cameras
/// back from python -- the whole result of the method. A python
/// implementation is called as `optimize( cameras, tracks, landmarks,
/// constraints )` and returns the camera map; the single-camera overload
/// is `optimize_camera` and returns the camera. See trampolines/README.md.

#ifndef OPTIMIZE_CAMERAS_TRAMPOLINE_TXX
#define OPTIMIZE_CAMERAS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <python/kwiver/vital/algo/algorithm_trampoline.txx>
#include <python/kwiver/vital/algo/out_parameter.txx>
#include <viame/algorithm_framework/algo/optimize_cameras.h>

namespace kwiver::vital::python {

template< class optimize_cameras_base = kwiver::vital::algo::optimize_cameras >
class optimize_cameras_trampoline
  : public algorithm_trampoline< optimize_cameras_base >
{
public:
  using algorithm_trampoline< optimize_cameras_base >::algorithm_trampoline;

  /// Optimize a map of cameras.
  ///
  /// `cameras` is in and out, and it is the entire result: the interface
  /// returns void. A python implementation returns the optimised map, and
  /// there is nothing else in the tuple, so the convention collapses to
  /// "return what you would have written into the parameter".
  void
  optimize(
    ::kwiver::vital::camera_map_sptr& cameras,
    ::kwiver::vital::feature_track_set_sptr tracks,
    ::kwiver::vital::landmark_map_sptr landmarks,
    ::kwiver::vital::sfm_constraints_sptr constraints ) const override
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< kwiver::vital::algo::optimize_cameras const* >( this ),
        "optimize" );

    if( !overload )
    {
      optimize_cameras_base::optimize(
        cameras, tracks, landmarks, constraints );
      return;
    }

    auto result = overload( cameras, tracks, landmarks, constraints );

    // A python implementation that returns nothing has optimised nothing
    // the caller can see, which is worth saying once rather than leaving
    // as a silent no-op.
    if( result.is_none() )
    {
      LOG_WARN(
        kwiver::vital::get_logger( "python.algo" ),
        "optimize_cameras.optimize returned None; the caller keeps the "
        "cameras it passed in, unoptimised." );
      return;
    }

    cameras = result.cast< kwiver::vital::camera_map_sptr >();
  }

  /// Optimize one camera against parallel feature and landmark vectors.
  ///
  /// Under a name of its own, `optimize_camera`, for the reason the
  /// estimators' second overload has one: python has a single name for the
  /// two C++ signatures, and an implementation that writes the map form
  /// would otherwise be called with this one's arguments.
  void
  optimize(
    ::kwiver::vital::camera_perspective_sptr& camera,
    ::std::vector< std::shared_ptr< kwiver::vital::feature > > const& features,
    ::std::vector< std::shared_ptr< kwiver::vital::landmark > > const& landmarks,
    ::kwiver::vital::sfm_constraints_sptr constraints ) const override
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< kwiver::vital::algo::optimize_cameras const* >( this ),
        "optimize_camera" );

    if( !overload )
    {
      pybind11::pybind11_fail(
        "Tried to call pure virtual function "
        "\"optimize_cameras::optimize\" (bound as optimize_camera)" );
    }

    auto result = overload( camera, features, landmarks, constraints );

    if( result.is_none() )
    {
      return;
    }

    camera = result.cast< kwiver::vital::camera_perspective_sptr >();
  }
};

} // namespace kwiver::vital::python

#undef KWIVER_PYBIND11_INCLUDE
#endif
