// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the core interpolate_track_spline algorithm
 */

#ifndef KWIVER_ARROWS_CORE_INTERPOLATE_TRACK_SPLINE_H_
#define KWIVER_ARROWS_CORE_INTERPOLATE_TRACK_SPLINE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/interpolate_track.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Fills in missing track segments using spline interpolation
/**
 * This class generates additional track states in between known states using
 * a configurable variety of spline-based interpolation techniques that do not
 * depend on imagery.
 */
class KWIVER_ALGO_CORE_EXPORT interpolate_track_spline
  : public vital::algo::interpolate_track
{
public:
  PLUGIN_INFO( "spline",
               "Fill in missing object track intervals using spline-based interpolation." )

  /// Default Constructor
  interpolate_track_spline();

  /// Destructor
  virtual ~interpolate_track_spline();

  /// Set this algo's properties via a config block
  virtual void set_configuration(
    vital::config_block_sptr config ) override;

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(
    vital::config_block_sptr config ) const override;

  /// Interpolates the states between track states
  virtual kwiver::vital::track_sptr interpolate(
    kwiver::vital::track_sptr init_states ) override;

protected:
  /// private implementation class
  class priv;
  std::unique_ptr<priv> const d_;
};

} } } // end namespace

#endif
