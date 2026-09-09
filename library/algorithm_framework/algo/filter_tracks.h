// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_FILTER_TRACKS_H_
#define VITAL_ALGO_FILTER_TRACKS_H_

#include <viame/algorithm_framework/vital_config.h>

#include <memory>
#include <utility>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/track_set.h>

/// \file
/// \brief Header defining abstract \link kwiver::vital::algo::filter_tracks
///        filter tracks \endlink algorithm

namespace kwiver {

namespace vital {

namespace algo {

/// \brief Abstract base class for track set filter algorithms.
class VITAL_ALGO_EXPORT filter_tracks
  : public kwiver::vital::algorithm
{
public:
  /// Return the name of this algorithm.
  filter_tracks();
  PLUGGABLE_INTERFACE( filter_tracks );
  /// Filter a track set and return a subset of the tracks
  ///
  /// \param [in] input The track set to filter
  /// \returns a filtered version of the track set (simple_track_set)
  virtual kwiver::vital::track_set_sptr
  filter( kwiver::vital::track_set_sptr input ) const = 0;
};

/// type definition for shared pointer to a filter_tracks algorithm
typedef std::shared_ptr< filter_tracks > filter_tracks_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_FILTER_TRACKS_H_
