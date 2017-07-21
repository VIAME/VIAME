/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Instantiation of \link kwiver::vital::algo::algorithm_def algorithm_def<T>
 *        \endlink for \link kwiver::vital::algo::optimize_cameras
 *        optimize_cameras \endlink
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/optimize_cameras.h>
#include <vital/vital_foreach.h>

namespace kwiver {
namespace vital {
namespace algo {

optimize_cameras
::optimize_cameras()
{
  attach_logger( "optimize_cameras" );
}


/// Optimize camera parameters given sets of landmarks and feature tracks
void
optimize_cameras
::optimize(camera_map_sptr & cameras,
           feature_track_set_sptr tracks,
           landmark_map_sptr landmarks,
           video_metadata_map_sptr metadata) const
{
  if (!cameras || !tracks || !landmarks)
  {
    throw invalid_value("One or more input data pieces are Null!");
  }
  typedef camera_map::map_camera_t map_camera_t;
  typedef landmark_map::map_landmark_t map_landmark_t;

  // extract data from containers
  map_camera_t cams = cameras->cameras();
  map_landmark_t lms = landmarks->landmarks();
  std::vector<track_sptr> trks = tracks->tracks();

  // Compose a map of frame IDs to a nested map of track ID to the state on
  // that frame number.
  typedef std::map< track_id_t, feature_sptr > inner_map_t;
  typedef std::map< frame_id_t, inner_map_t > states_map_t;

  states_map_t states_map;
  // O( len(trks) * avg_t_len )
  VITAL_FOREACH(track_sptr const& t, trks)
  {
    // Only record a state if there is a corresponding landmark for the
    // track (constant-time check), the track state has a feature and thus a
    // location (constant-time check), and if we have a camera on the track
    // state's frame (constant-time check).
    if (lms.count(t->id()))
    {
      VITAL_FOREACH (auto const& ts, *t)
      {
        auto fts = std::dynamic_pointer_cast<feature_track_state>(ts);
        if (fts && fts->feature && cams.count(ts->frame()))
        {
          states_map[ts->frame()][t->id()] = fts->feature;
        }
      }
    }
  }

  // For each camera in the input map, create corresponding point sets for 2D
  // and 3D coordinates of tracks and matching landmarks, respectively, for
  // that camera's frame.
  map_camera_t optimized_cameras;
  std::vector< feature_sptr > v_feat;
  std::vector< landmark_sptr > v_lms;
  video_metadata_map::map_video_metadata_t metadata_map;
  if(metadata)
  {
    metadata_map = metadata->video_metadata();
  }
  VITAL_FOREACH(map_camera_t::value_type const& p, cams)
  {
    v_feat.clear();
    v_lms.clear();
    video_metadata_vector v_metadata;

    auto mdv = metadata_map.find(p.first);
    if(mdv != metadata_map.end())
    {
      v_metadata = mdv->second;
    }

    // Construct 2d<->3d correspondences
    VITAL_FOREACH(inner_map_t::value_type const& q, states_map[p.first])
    {
      // Already guaranteed that feat and landmark exists above.
      v_feat.push_back(q.second);
      v_lms.push_back(lms[q.first]);
    }

    camera_sptr cam = p.second;
    this->optimize(cam, v_feat, v_lms, v_metadata);
    optimized_cameras[p.first] = cam;
  }

  cameras = camera_map_sptr(new simple_camera_map(optimized_cameras));
}


} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::optimize_cameras);
/// \endcond
