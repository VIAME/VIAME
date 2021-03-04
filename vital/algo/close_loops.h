// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_CLOSE_LOOPS_H_
#define VITAL_ALGO_CLOSE_LOOPS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>

#include <ostream>

/**
 * \file
 * \brief Header defining abstract \link kwiver::vital::algo::close_loops
 *        close_loops \endlink algorithm
 */

namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for loop closure algorithms.
/**
 * Different algorithms can perform loop closure in a variety of ways, either
 * in attempt to make either short or long term closures. Similarly to
 * track_features, this class is designed to be called in an online fashion.
 */
class VITAL_ALGO_EXPORT close_loops
  : public kwiver::vital::algorithm_def<close_loops>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "close_loops"; }

  /// Attempt to perform closure operation and stitch tracks together.
  /**
   * \param frame_number the frame number of the current frame
   * \param input the input feature track set to stitch
   * \param image image data for the current frame
   * \param mask Optional mask image where positive values indicate
   *                  regions to consider in the input image.
   * \returns an updated set of feature tracks after the stitching operation
   */
  virtual kwiver::vital::feature_track_set_sptr
  stitch( kwiver::vital::frame_id_t frame_number,
          kwiver::vital::feature_track_set_sptr input,
          kwiver::vital::image_container_sptr image,
          kwiver::vital::image_container_sptr mask = kwiver::vital::image_container_sptr()) const = 0;

protected:
  close_loops();

};

typedef std::shared_ptr<close_loops> close_loops_sptr;

} } } // end namespace

#endif // VITAL_ALGO_CLOSE_LOOPS_H_
