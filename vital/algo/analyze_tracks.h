// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_ANALYZE_TRACKS_H_
#define VITAL_ALGO_ANALYZE_TRACKS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/track_set.h>

#include <ostream>
#include <memory>

/**
 * \file
 * \brief Header defining abstract \link kwiver::vital::algo::analyze_tracks track
 *        analyzer \endlink algorithm
 */

namespace kwiver {
namespace vital {
namespace algo {

/// Abstract base class for writing out human readable track statistics.
class VITAL_ALGO_EXPORT analyze_tracks
  : public kwiver::vital::algorithm_def<analyze_tracks>
{
public:

  typedef std::ostream stream_t;

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "analyze_tracks"; }

  /// Output various information about the tracks stored in the input set.
  /**
   * \param [in] track_set the tracks to analyze
   * \param [in] stream an output stream to write data onto
   */
  virtual void
  print_info(kwiver::vital::track_set_sptr track_set,
             stream_t& stream = std::cout) const = 0;

protected:
  analyze_tracks();

};

typedef std::shared_ptr<analyze_tracks> analyze_tracks_sptr;

} } } // end namespace

#endif // VITAL_ALGO_ANALYZE_TRACKS_H_
