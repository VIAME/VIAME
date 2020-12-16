// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for OpenCV analyze_tracks algorithm
 */

#ifndef KWIVER_ARROWS_OCV_ANALYZE_TRACKS_H_
#define KWIVER_ARROWS_OCV_ANALYZE_TRACKS_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/analyze_tracks.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class for outputting various debug info about feature tracks
class KWIVER_ALGO_OCV_EXPORT analyze_tracks
: public vital::algo::analyze_tracks
{
public:
  PLUGIN_INFO( "ocv",
               "Use OpenCV to analyze statistics of feature tracks." )

  /// Constructor
  analyze_tracks();

  /// Destructor
  virtual ~analyze_tracks();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Output various information about the tracks stored in the input set.
  /**
   * \param [in] track_set the tracks to analyze
   * \param [in] stream an output stream to write data onto
   */
  virtual void
  print_info(vital::track_set_sptr track_set,
             stream_t& stream = std::cout) const;

private:

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
