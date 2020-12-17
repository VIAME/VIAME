// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_FILTER_TRACKS_H_
#define VITAL_ALGO_FILTER_TRACKS_H_

#include <vital/vital_config.h>

#include <utility>
#include <vector>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/track_set.h>

/**
 * \file
 * \brief Header defining abstract \link kwiver::vital::algo::filter_tracks
 *        filter tracks \endlink algorithm
 */

namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for track set filter algorithms.
class VITAL_ALGO_EXPORT filter_tracks
  : public kwiver::vital::algorithm_def<filter_tracks>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "filter_tracks"; }

  /// Filter a track set and return a subset of the tracks
  /**
   * \param [in] input The track set to filter
   * \returns a filtered version of the track set (simple_track_set)
   */
  virtual kwiver::vital::track_set_sptr
  filter( kwiver::vital::track_set_sptr input ) const = 0;

protected:
  filter_tracks();

};

/// type definition for shared pointer to a filter_tracks algorithm
typedef std::shared_ptr<filter_tracks> filter_tracks_sptr;

} } } // end namespace

#endif // VITAL_ALGO_FILTER_TRACKS_H_
