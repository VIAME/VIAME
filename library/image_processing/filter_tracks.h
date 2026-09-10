// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CORE_FILTER_TRACKS_H_
#define KWIVER_ARROWS_CORE_FILTER_TRACKS_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/filter_tracks.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

/// \file
/// \brief Header defining the core filter_tracks algorithm

namespace kwiver {

namespace arrows {

namespace core {

/// \brief Algorithm that filters tracks on various attributes
class VIAME_IMAGE_PROCESSING_EXPORT filter_tracks
  : public vital::algo::filter_tracks
{
public:
  PLUGGABLE_IMPL(
    filter_tracks,
    "Filter tracks by track length or matrix matrix importance.",
    PARAM_DEFAULT(
      min_track_length, size_t,
      "Filter the tracks keeping those covering "
      "at least this many frames. Set to 0 to disable.",
      3 ),
    PARAM_DEFAULT(
      min_mm_importance, double,
      "Filter the tracks with match matrix importance score "
      "below this threshold. Set to 0 to disable.",
      1.0 )
  )

  /// Destructor
  virtual ~filter_tracks();

  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// filter a track set
  ///
  /// \param track set to filter
  /// \returns a filtered version of the track set
  virtual vital::track_set_sptr
  filter( vital::track_set_sptr input ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif
