// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of VXL triangulate landmarks algorithm
 */

#include "triangulate_landmarks.h"

#include <set>

#include <vital/vital_config.h>

#include <arrows/vxl/camera_map.h>

#include <vpgl/algo/vpgl_triangulate_points.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

// Private implementation class
class triangulate_landmarks::priv
{
public:
  // Constructor
  priv()
  {
  }

  // parameters - none yet
};

// ----------------------------------------------------------------------------
// Constructor
triangulate_landmarks
::triangulate_landmarks()
: d_(new priv)
{
  attach_logger( "arrows.vxl.triangulate_landmarks" );
}

// Destructor
triangulate_landmarks
::~triangulate_landmarks()
{
}

// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
triangulate_landmarks
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::triangulate_landmarks::get_configuration();
  return config;
}

// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
triangulate_landmarks
::set_configuration( VITAL_UNUSED vital::config_block_sptr in_config)
{
}

// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
triangulate_landmarks
::check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
{
  return true;
}

// ----------------------------------------------------------------------------
// Triangulate the landmark locations given sets of cameras and feature tracks
void
triangulate_landmarks
::triangulate(vital::camera_map_sptr cameras,
              vital::feature_track_set_sptr tracks,
              vital::landmark_map_sptr& landmarks) const
{
  if( !cameras || !landmarks || !tracks )
  {
    // TODO throw an exception for missing input data
    return;
  }
  typedef vxl::camera_map::map_vcam_t map_vcam_t;
  typedef vital::landmark_map::map_landmark_t map_landmark_t;

  // extract data from containers
  map_vcam_t vcams = camera_map_to_vpgl(*cameras);
  map_landmark_t lms = landmarks->landmarks();
  std::vector<track_sptr> trks = tracks->tracks();

  // build a track map by id
  typedef std::map<track_id_t, track_sptr> track_map_t;
  track_map_t track_map;
  for(const track_sptr& t : trks)
  {
    track_map[t->id()] = t;
  }

  // the set of landmark ids which failed to triangulation
  std::set<landmark_id_t> failed_landmarks;

  map_landmark_t triangulated_lms;
  for(const map_landmark_t::value_type& p : lms)
  {
    // get the corresponding track
    track_map_t::const_iterator t_itr = track_map.find(p.first);
    if (t_itr == track_map.end())
    {
      // there is no track for the provided landmark
      continue;
    }
    const track& t = *t_itr->second;

    // extract the cameras and image points for this landmarks
    std::vector<vpgl_perspective_camera<double> > lm_cams;
    std::vector<vgl_point_2d<double> > lm_image_pts;
    std::vector<feature_track_state_sptr> feats;

    for (track::history_const_itr tsi = t.begin(); tsi != t.end(); ++tsi)
    {
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*tsi);
      if (!fts || !fts->feature)
      {
        // there is no valid feature for this track state
        continue;
      }
      map_vcam_t::const_iterator c_itr = vcams.find((*tsi)->frame());
      if (c_itr == vcams.end())
      {
        // there is no camera for this track state.
        continue;
      }
      lm_cams.push_back(c_itr->second);
      vital::vector_2d pt = fts->feature->loc();
      lm_image_pts.push_back(vgl_point_2d<double>(pt.x(), pt.y()));
      feats.push_back(fts);
    }

    // if we found at least two views of this landmark, triangulate
    if (lm_cams.size() > 1)
    {
      vital::vector_3d lm_loc = p.second->loc();
      vgl_point_3d<double> pt3d(lm_loc.x(), lm_loc.y(), lm_loc.z());
      double error = vpgl_triangulate_points::triangulate(lm_image_pts,
                                                          lm_cams, pt3d);
      bool bad_triangulation = false;
      vgl_homg_point_3d<double> hpt3d(pt3d);
      for(vpgl_perspective_camera<double> const& cam : lm_cams)
      {
        if(cam.is_behind_camera(hpt3d))
        {
          bad_triangulation = true;
          failed_landmarks.insert(p.first);
          break;
        }
      }
      if( !bad_triangulation )
      {
        auto lm = std::make_shared<vital::landmark_d>();
        lm->set_loc(vital::vector_3d(pt3d.x(), pt3d.y(), pt3d.z()));
        lm->set_covar(covariance_3d(error));
        lm->set_observations(static_cast<unsigned int>(lm_cams.size()));
        triangulated_lms[p.first] = lm;
        for (auto fts : feats)
        {
          fts->inlier = true;
        }
      }
    }
  }
  if( !failed_landmarks.empty() )
  {
    LOG_ERROR(logger(), "failed to triangulate " << failed_landmarks.size()
                            << " of " << lms.size() << " landmarks");
  }
  landmarks = landmark_map_sptr(new simple_landmark_map(triangulated_lms));
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
