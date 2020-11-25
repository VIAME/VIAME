// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of map from frame IDs to vpgl cameras
 */

#include "camera_map.h"

#include <arrows/vxl/camera.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

/// Return a map from integer IDs to camera shared pointers
vital::camera_map::map_camera_t
camera_map::cameras() const
{
  map_camera_t vital_cameras;

  for(const map_vcam_t::value_type& c : data_)
  {
    camera_sptr cam = vpgl_camera_to_vital(c.second);
    vital_cameras.insert(std::make_pair(c.first, cam));
  }

  return vital_cameras;
}

/// Convert any camera map to a vpgl camera map
camera_map::map_vcam_t
camera_map_to_vpgl(const vital::camera_map& cam_map)
{
  // if the camera map already contains a vpgl representation
  // then return the existing vpgl data
  if( const vxl::camera_map* m =
          dynamic_cast<const vxl::camera_map*>(&cam_map) )
  {
    return m->vpgl_cameras();
  }
  camera_map::map_vcam_t vmap;
  for (const camera_map::map_camera_t::value_type& c :
                cam_map.cameras())
  {
    vpgl_perspective_camera<double> vcam;
    if( const simple_camera_perspective* mcam =
        dynamic_cast<const simple_camera_perspective*>(c.second.get()) )
    {
      vital_to_vpgl_camera(*mcam, vcam);
    }
    else
    {
      //TODO should throw an exception here
    }
    vmap.insert(std::make_pair(c.first, vcam));
  }
  return vmap;
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
