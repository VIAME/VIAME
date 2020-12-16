// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for OCV draw_tracks algorithm
 */

#ifndef KWIVER_ARROWS_OCV_DRAW_TRACKS_H_
#define KWIVER_ARROWS_OCV_DRAW_TRACKS_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/draw_tracks.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class for drawing various information about feature tracks
class KWIVER_ALGO_OCV_EXPORT draw_tracks
: public vital::algo::draw_tracks
{
public:
  PLUGIN_INFO( "ocv",
               "Use OpenCV to draw tracked features on the images." )

  /// Constructor
  draw_tracks();

  /// Destructor
  virtual ~draw_tracks();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Draw features tracks on top of the input images.
  /**
   * This process can either be called in an offline fashion, where all
   * tracks and images are provided to the function on the first call,
   * or in an online fashion where only new images are provided on
   * sequential calls. This function can additionally consumes a second
   * track set for which can optionally be used to display additional
   * information to provide a comparison between the two track sets.
   *
   * \param [in] display_set the main track set to draw
   * \param [in] image_data a list of images the tracks were computed over
   * \param [in] comparison_set optional comparison track set
   * \returns a pointer to the last image generated
   */
  virtual vital::image_container_sptr
  draw(vital::track_set_sptr display_set,
       vital::image_container_sptr_list image_data,
       vital::track_set_sptr comparison_set = vital::track_set_sptr());

private:

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
