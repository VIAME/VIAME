// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief projected_track_set implementation
 */

#include "projected_track_set.h"

#include <vital/types/feature.h>

namespace kwiver {
namespace arrows {
namespace mvg {

using namespace kwiver::vital;

/// Use the cameras to project the landmarks back into their images.
feature_track_set_sptr
projected_tracks(landmark_map_sptr landmarks, camera_map_sptr cameras)
{
  std::vector<track_sptr> tracks;

  camera_map::map_camera_t cam_map = cameras->cameras();
  landmark_map::map_landmark_t lm_map = landmarks->landmarks();

  for( landmark_map::map_landmark_t::iterator l = lm_map.begin(); l != lm_map.end(); l++ )
  {
    track_sptr t = track::create();
    t->set_id( l->first );
    tracks.push_back( t );

    for( const camera_map::map_camera_t::value_type& p : cam_map )
    {
      const camera_sptr cam = p.second;
      auto fts = std::make_shared<feature_track_state>(p.first);
      fts->feature = std::make_shared<feature_d>( cam->project( l->second->loc() ) );
      fts->inlier = true;
      t->append( fts );
    }
  }
  return std::make_shared<feature_track_set>( tracks );
}

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver
