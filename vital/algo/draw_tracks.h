// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_DRAW_TRACKS_H_
#define VITAL_ALGO_DRAW_TRACKS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/track_set.h>

#include <ostream>

/**
 * \file
 * \brief Header defining an abstract \link kwiver::vital::algo::draw_tracks track
 *        drawing \endlink algorithm
 */

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for algorithms which draw tracks on top of
/// images in various ways, for analyzing results.
class VITAL_ALGO_EXPORT draw_tracks
  : public kwiver::vital::algorithm_def<draw_tracks>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "draw_tracks"; }

  /// Draw features tracks on top of the input images.
  /**
   * This process can either be called in an offline fashion, where all
   * tracks and images are provided to the function on the first call,
   * or in an online fashion where only new images are provided on
   * sequential calls. This function can additionally consume a second
   * track set, which can optionally be used to display additional
   * information to provide a comparison between the two track sets.
   *
   * \param display_set the main track set to draw
   * \param image_data a list of images the tracks were computed over
   * \param comparison_set optional comparison track set
   * \returns a pointer to the last image generated
   */
  virtual kwiver::vital::image_container_sptr
  draw(kwiver::vital::track_set_sptr display_set,
       kwiver::vital::image_container_sptr_list image_data,
       kwiver::vital::track_set_sptr comparison_set = kwiver::vital::track_set_sptr()) = 0;

protected:
    draw_tracks();

};

/// A smart pointer to a draw_tracks instance.
typedef std::shared_ptr<draw_tracks> draw_tracks_sptr;

} } } // end namespace algo

#endif // VITAL_ALGO_DRAW_TRACKS_H_
