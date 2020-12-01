// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CORE_FILTER_TRACKS_H_
#define KWIVER_ARROWS_CORE_FILTER_TRACKS_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/filter_tracks.h>

/**
 * \file
 * \brief Header defining the core filter_tracks algorithm
 */

namespace kwiver {
namespace arrows {
namespace core {

/// \brief Algorithm that filters tracks on various attributes
class KWIVER_ALGO_CORE_EXPORT filter_tracks
  : public vital::algo::filter_tracks
{
public:
  PLUGIN_INFO( "core",
               "Filter tracks by track length or matrix matrix importance." )

  /// Constructor
  filter_tracks();

  /// Destructor
  virtual ~filter_tracks();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// filter a track set
  /**
   * \param track set to filter
   * \returns a filtered version of the track set
   */
  virtual vital::track_set_sptr
  filter(vital::track_set_sptr input) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
