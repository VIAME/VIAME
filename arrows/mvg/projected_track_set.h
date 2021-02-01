// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for functions relating to generating projected
 * tracks from a sequence of landmarks and camera parameters.
 */

#ifndef KWIVER_ARROWS_MVG_PROJECTED_TRACK_SET_H_
#define KWIVER_ARROWS_MVG_PROJECTED_TRACK_SET_H_

#include <vital/vital_config.h>
#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/types/feature_track_set.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// Use the cameras to project the landmarks back into their images.
/**
 * \param landmarks input landmark locations
 * \param cameras input camera map
 * \return feature track set generated via the projection
 */
vital::feature_track_set_sptr
KWIVER_ALGO_MVG_EXPORT
projected_tracks(vital::landmark_map_sptr landmarks,
                 vital::camera_map_sptr cameras);

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif // ALGORITHMS_PROJECTED_TRACK_SET_H_
